"""审查发现的问题：无效干预、验证目标、配置范围、预算与源链接。"""
import copy
import tempfile
import unittest
from pathlib import Path

from experiments.drivers.sft import evaluation_config, validate_config as validate_sft
from experiments.drivers.vision import rate_at, validate_config
from experiments.tools.build_catalog import controls_for, render_wikilinks
from experiments.utils.runtime import validate_splits, build_plan, read_json, ROOT, preflight_native
from experiments.utils.training_checks import (
    validate_active_axis, EvaluationClock, require_complete_epoch,
)


class ReviewRegressions(unittest.TestCase):
    def test_inert_scheduler_and_optimizer_parameters_rejected(self):
        for axis,value in [('optimizer.gamma',.9),('optimizer.warmup_ratio',.1),
                           ('optimizer.min_lr',.001),('optimizer.beta2',.95)]:
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                validate_active_axis({axis:value},axis)

    def test_registered_scheduler_contrasts_really_change_rates(self):
        for axis,a,b in [('optimizer.gamma',.5,.99),('optimizer.warmup_ratio',.03,.1),
                         ('optimizer.min_lr',0.,.00001)]:
            controls=controls_for({'id':'unit-fixture'},axis)
            config_a={**controls,axis:a,'optimizer.lr':.1}
            config_b={**controls,axis:b,'optimizer.lr':.1}
            validate_active_axis(config_a,axis)
            self.assertNotEqual([rate_at(s,100,config_a,1.) for s in range(100)],
                                [rate_at(s,100,config_b,1.) for s in range(100)])

    def test_validation_target_does_not_follow_training_mask(self):
        a={'loss.assistant_only':False,'data.max_length':512}
        b={'loss.assistant_only':True,'data.max_length':512}
        self.assertEqual(evaluation_config(a),evaluation_config(b))
        self.assertFalse(a['loss.assistant_only'])

    def test_evaluation_cadence_and_final_tail(self):
        clock=EvaluationClock(8)
        self.assertEqual([clock.due(n) for n in (4,8,12,16)],[False,True,False,True])
        self.assertFalse(clock.due(16,final=True))
        self.assertTrue(clock.due(17,final=True))
        with self.assertRaises(ValueError):
            clock.due(10)

    def test_invalid_configuration_is_rejected_before_training(self):
        for config in [{'training.batch_size':0},{'training.epochs':False},
                       {'optimizer.lr':float('nan')},{'model.dropout':1.},
                       {'model.norm':'weight'},{'training.precision':'fp8'},
                       {'data.shuffle':'false'}]:
            with self.subTest(config=config),self.assertRaises(ValueError):
                validate_config(config)
        for config in [{'data.max_length':0},{'adapter.rank':-1},{'adapter.target_modules':[]}]:
            with self.subTest(config=config),self.assertRaises(ValueError):
                validate_sft(config)
        with self.assertRaises(ValueError):
            validate_sft({},'typo')

    def test_native_preflight_checks_both_arms_and_metric(self):
        spec=read_json(ROOT/'experiments/activation/activation-001-01/experiment.json')
        profile=read_json(ROOT/'experiments/configs/remote.example.json')['profiles']['vision']
        altered=copy.deepcopy(spec)
        altered['b']='unknown_activation'
        with self.assertRaises(ValueError):
            preflight_native(profile,altered)
        altered=copy.deepcopy(spec)
        altered['metric']='test_map'
        with self.assertRaises(ValueError):
            preflight_native(profile,altered)

    def test_bad_selector_rejected_even_when_sources_differ(self):
        for selector in [None,{'source':'train','modulo':0,'remainder':0},
                         {'source':'train','modulo':3,'remainder':3},
                         {'source':'train','modulo':3,'remainder':0,'keep_equal':'false'}]:
            with self.subTest(selector=selector),self.assertRaises(ValueError):
                validate_splits({'train':selector,'validation':{'source':'val'},'test':{'source':'test'}})

    def test_truncated_epoch_cannot_be_reported_complete(self):
        with self.assertRaises(RuntimeError):
            require_complete_epoch(40,45)
        require_complete_epoch(45,45)

    def test_bad_seed_cannot_reach_numpy(self):
        spec=read_json(ROOT/'experiments/activation/activation-001-01/experiment.json')
        for seeds in ([1,-1],[1,2**32],[True,2]):
            with self.subTest(seeds=seeds),self.assertRaises(ValueError):
                build_plan(spec,seeds)

    def test_wikilinks_are_resolved_from_source_directory(self):
        source=ROOT/'wiki/activation/activation.md'
        target=ROOT/'experiments/activation/activation-001-01'
        rendered=render_wikilinks('[[relu|解释]] [[missing-paper]]',source,target)
        self.assertIn('[解释](../../../wiki/activation/relu.md)',rendered)
        self.assertIn('原 Wiki 未解析链接',rendered)


if __name__ == '__main__':
    unittest.main()
