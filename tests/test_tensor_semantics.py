"""仅 CPU 数学夹具；不下载权重，不生成可被当作训练结论的产物。"""
import copy
import importlib.util
import re
import unittest

from experiments.utils.runtime import ROOT

TORCH_AVAILABLE=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE,'数值回归需要 torch；CI 必须安装 CPU torch 执行')
class TensorSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_sgd_l2_enters_momentum_and_differs_from_decoupled_decay(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9, weight_decay=.2)
        expected, buffer, decoupled = 1., 0., 1.
        for _ in range(2):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            buffer = .9 * buffer + .2 * expected
            expected -= .1 * buffer
            decoupled *= 1 - .1 * .2
        self.assertAlmostEqual(parameter.item(), expected)
        self.assertNotAlmostEqual(parameter.item(), decoupled)

    def test_wiki_accumulation_code_matches_full_batches_with_tail(self):
        import torch
        torch.manual_seed(17)
        model=torch.nn.Linear(3,2).double()
        reference=copy.deepcopy(model)
        rows=[(torch.randn(n,3,dtype=torch.double),torch.arange(n)%2) for n in (2,3,1,2,1)]
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        reference_optimizer=torch.optim.SGD(reference.parameters(),lr=.1)
        wiki=(ROOT/'wiki/batch-size/gradient-accumulation.md').read_text()
        snippet=re.search(r'```python\n(.*?)```',wiki,re.S)[1]
        exec(snippet,{'model':model,'dataloader':rows,'optimizer':optimizer,'device':'cpu'})
        for group in (rows[:4],rows[4:]):
            reference_optimizer.zero_grad()
            images=torch.cat([x for x,_ in group])
            labels=torch.cat([y for _,y in group])
            torch.nn.functional.cross_entropy(reference(images),labels).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
            reference_optimizer.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)

    def test_reply_token_weighting_matches_combined_loss(self):
        import torch
        torch.manual_seed(42)
        logits=torch.randn(2,4,7,dtype=torch.double,requires_grad=True)
        labels=torch.tensor([[1,2,-100,-100],[2,3,4,5]])
        combined=torch.nn.functional.cross_entropy(logits.reshape(-1,7),labels.reshape(-1))
        expected=torch.autograd.grad(combined,logits,retain_graph=True)[0]
        weighted=sum(torch.nn.functional.cross_entropy(logits[i],labels[i])*(labels[i]!=-100).sum()/6 for i in range(2))
        actual=torch.autograd.grad(weighted,logits)[0]
        torch.testing.assert_close(actual,expected)

    def test_resnet_shapes_and_gradients(self):
        import torch
        from experiments.drivers.vision_model import ResNet
        for norm in ('none','batch','layer','group','instance'):
            with self.subTest(norm=norm):
                model=ResNet({'model.width':8,'model.depth':1,'model.norm':norm})
                outputs=model(torch.randn(2,3,32,32))
                self.assertEqual(tuple(outputs.shape),(2,10))
                outputs.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_default_initialization_is_not_he_normal_alias(self):
        import torch
        from experiments.drivers.vision_model import ResNet
        torch.manual_seed(17)
        a=ResNet({'model.width':8,'model.depth':1,'model.weight_init':'default'})
        torch.manual_seed(17)
        b=ResNet({'model.width':8,'model.depth':1,'model.weight_init':'kaiming_normal'})
        self.assertFalse(torch.equal(a.stem[0].weight,b.stem[0].weight))

    def test_optimizer_step_actually_changes_model(self):
        import torch
        from experiments.drivers.vision import optimizer_for,criterion
        for kind in ('sgd','adam','adamw','adagrad'):
            model=torch.nn.Linear(3,2)
            before=model.weight.detach().clone()
            optimizer=optimizer_for(model,{'optimizer.kind':kind,'optimizer.lr':.01})
            criterion(model(torch.ones(3,3)),torch.tensor([0,1,0]),{}).mean().backward()
            optimizer.step()
            self.assertFalse(torch.equal(before,model.weight))


if __name__ == '__main__':
    unittest.main()
