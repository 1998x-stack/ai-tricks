"""离线契约测试。所有极小数据均为测试夹具，不是实验训练结果。"""
import copy
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.drivers.sft import tokenize_example, train as train_sft
from experiments.drivers.vision import train as train_vision, validate_config
from experiments.utils.hf_stream import ids_digest, require_remote
from experiments.utils.reporting import plot_results, summarize
from experiments.utils.runtime import (
    ROOT, build_plan, check_pairs, check_remote, file_hash, read_json,
    validate_result, validate_splits, write_json,
)


class ExperimentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(ROOT/'experiments/activation/activation-001-01/experiment.json')

    def test_plan_has_paired_seeds_and_alternating_order(self):
        plan = build_plan(self.spec,[17,42,2026])
        self.assertEqual([(r['seed'],r['arm']) for r in plan['runs']],
                         [(17,'A'),(17,'B'),(42,'B'),(42,'A'),(2026,'A'),(2026,'B')])
        for run in plan['runs']:
            self.assertEqual(list(run['intervention']),[self.spec['axis']])

    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):
            build_plan(self.spec,[42,42])

    def test_plan_hash_changes_with_intervention(self):
        alternate = copy.deepcopy(self.spec)
        alternate['b'] = 'tanh'
        self.assertNotEqual(build_plan(self.spec,[1,2])['plan_hash'],
                            build_plan(alternate,[1,2])['plan_hash'])

    def test_non_remote_host_rejected_before_any_dataset_access(self):
        environment = {'execution':{'allowed_hostnames':[]}}
        with self.assertRaises(RuntimeError):
            check_remote(environment,self.spec)
        with self.assertRaises(RuntimeError):
            require_remote({'allowed_hostnames':[]})
        for trainer in (train_vision,train_sft):
            with self.assertRaises(RuntimeError):
                trainer({'profile':{'allowed_hostnames':[]},'config':{}},Path('/not-created'))

    def test_split_overlap_rejected_even_with_different_selectors(self):
        bad = {'train':{'source':'train','modulo':2,'remainder':0},
               'validation':{'source':'train','modulo':4,'remainder':0},
               'test':{'source':'test'}}
        with self.assertRaises(ValueError):
            validate_splits(bad)
        good = copy.deepcopy(bad)
        good['validation'] = {'source':'train','modulo':2,'remainder':1}
        validate_splits(good)

    def test_unknown_intervention_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            validate_config({'model.magic':True})
        with self.assertRaises(ValueError):
            validate_config({'data.augmentation':'cutmix'})

    def test_unpaired_and_mismatched_test_sets_rejected(self):
        rows = [{'seed':1,'arm':'A','test_ids_sha256':'a'*64}]
        with self.assertRaises(ValueError):
            check_pairs(rows)
        rows.append({'seed':1,'arm':'B','test_ids_sha256':'b'*64})
        with self.assertRaises(ValueError):
            check_pairs(rows)

    def fixture_result(self,directory):
        request = {'experiment_id':'unit_fixture','plan_hash':'test_only','seed':1,'arm':'A',
                   'dataset_revision':'a'*40,'model_revision':'b'*40,
                   'config':{'optimizer.lr':.01},'metric':'test_accuracy'}
        checkpoint = directory/'checkpoint.fixture'
        checkpoint.write_text('contract test only; not model weights')
        write_json(directory/'checkpoint_manifest.json',{'selected_by_validation':True,
                   'path':checkpoint.name,'sha256':file_hash(checkpoint),'step':1})
        (directory/'history.csv').write_text('step,train_loss,val_loss\n1,1.0,1.2\n')
        (directory/'predictions.jsonl').write_text('{"id":"test:1","prediction":0}\n')
        result = {key:request[key] for key in ('experiment_id','plan_hash','seed','arm','dataset_revision','model_revision')}
        result.update({'applied_config':request['config'],'data_kind':'huggingface_real',
                       'train_examples_seen':1,'optimizer_steps':1,'metrics':{'test_accuracy':.5},
                       'test_ids_sha256':ids_digest(['test:1'])})
        return request,result

    def test_artifact_fingerprint_and_applied_config_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            request,result = self.fixture_result(path)
            validate_result(result,request,path)
            invalid = copy.deepcopy(result)
            invalid['applied_config']['optimizer.lr'] = .1
            with self.assertRaises(ValueError):
                validate_result(invalid,request,path)
            (path/'checkpoint.fixture').write_text('tampered')
            with self.assertRaises(ValueError):
                validate_result(result,request,path)

    def test_history_and_checkpoint_steps_cannot_claim_missing_updates(self):
        for checkpoint_step, history in [
            (0, '1,1,1\n'), (2, '1,1,1\n'),
            (1, '1,1,1\n1,1,1\n'), (1, '2,1,1\n'),
        ]:
            with self.subTest(step=checkpoint_step, history=history), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                request, result = self.fixture_result(path)
                manifest = read_json(path/'checkpoint_manifest.json')
                manifest['step'] = checkpoint_step
                write_json(path/'checkpoint_manifest.json', manifest)
                (path/'history.csv').write_text('step,train_loss,val_loss\n'+history)
                with self.assertRaises(ValueError):
                    validate_result(result, request, path)

    def test_runner_marks_training_and_reporting_failures(self):
        from types import SimpleNamespace
        path = ROOT/'experiments/activation/activation-001-01/experiment.py'
        spec = importlib.util.spec_from_file_location('failure_entry_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = {'environment': {'execution': {'max_total_seconds': 100, 'max_run_seconds': 20}},
                'runs': [{'seed': 1, 'arm': 'A'}], 'spec': {}}
        for failure_point in ('execute_run', 'summarize'):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'run'
                with patch.object(module, 'provenance', return_value={}), \
                     patch.object(module, 'execute_run', return_value={}), \
                     patch.object(module, failure_point, side_effect=RuntimeError('fixture failure')):
                    with self.assertRaises(RuntimeError):
                        module.run_all(SimpleNamespace(output=output), plan, {})
                status = read_json(output/'status.json')
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['completed_runs'], 0 if failure_point == 'execute_run' else 1)

    def test_prediction_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            request,result=self.fixture_result(path)
            (path/'predictions.jsonl').write_text('{"id":"different"}\n')
            with self.assertRaises(ValueError):
                validate_result(result,request,path)

    def test_nonfinite_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            request,result=self.fixture_result(path)
            result['metrics']['test_accuracy']=float('nan')
            with self.assertRaises(ValueError):
                validate_result(result,request,path)

    def test_generated_entry_runs_plan_without_network(self):
        path=ROOT/'experiments/activation/activation-001-01/experiment.py'
        spec=importlib.util.spec_from_file_location('entry_fixture',path)
        module=importlib.util.module_from_spec(spec)
        with patch.object(socket,'create_connection',side_effect=AssertionError('network forbidden')):
            spec.loader.exec_module(module)
            plan=build_plan(module.load_protocol(),[17,42])
        self.assertEqual(len(plan['runs']),4)

    def test_sft_mask_includes_only_assistant_and_preserves_truncated_reply(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
                ids=[]
                for message in messages:
                    ids += [1 if message['role']=='user' else 2] + list(message['content']) + [3]
                return ids + ([2] if add_generation_prompt else [])
        row={'messages':[{'role':'user','content':[10,11]}, {'role':'assistant','content':[20,21]}]}
        ids,labels=tokenize_example(row,Tokenizer(),{'data.max_length':4},'messages')
        self.assertEqual(ids,[2,20,21,3])
        self.assertEqual(labels,[-100,20,21,3])
        _,all_labels=tokenize_example(row,Tokenizer(),{'data.max_length':4,'loss.assistant_only':False},'messages')
        self.assertEqual(all_labels,ids)

    def test_reporting_fixture_keeps_negative_effect_and_writes_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for seed in (1,2,3):
                for arm,value in (('A',.8),('B',.7)):
                    rows.append({'seed':seed,'arm':arm,'metrics':{'test_accuracy':value},
                                 'test_ids_sha256':'a'*64})
                    directory=Path(tmp)/f'seed-{seed}'/arm
                    directory.mkdir(parents=True)
                    (directory/'history.csv').write_text('step,train_loss,val_loss\n1,1.0,1.1\n2,.8,.9\n')
            summary=summarize(rows,self.spec,tmp)
            self.assertAlmostEqual(summary['mean_delta'],-.1)
            plot_results(rows,self.spec,tmp)
            self.assertGreater((Path(tmp)/'comparison.png').stat().st_size,1000)
            self.assertGreater((Path(tmp)/'comparison.svg').stat().st_size,1000)


if __name__ == '__main__':
    unittest.main()
