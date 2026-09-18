import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from fluxbin_style.acceleration_checks import numerical_gate
from fluxbin_style.engine_interface import M1Backend

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_image',ROOT/'scripts/prepare_acceleration_image.py')
image=importlib.util.module_from_spec(spec);spec.loader.exec_module(image)


class PreparationTests(unittest.TestCase):
    def test_numerical_gate_zero_nonfinite_and_wrong_output(self):
        ref=torch.zeros(1,7,dtype=torch.bfloat16)
        self.assertTrue(numerical_gate(ref,ref)['passed'])
        self.assertFalse(numerical_gate(torch.ones_like(ref),ref)['passed'])
        self.assertFalse(numerical_gate(ref+float('nan'),ref)['passed'])
        self.assertFalse(numerical_gate(torch.zeros(2,7),ref)['passed'])

    def test_engine_contract_no_serving_dependency(self):
        backend=M1Backend()
        self.assertEqual(backend.contract.supported_m,(1,))
        self.assertFalse(backend.contract.tensor_parallel)
        self.assertEqual(backend.workspace_shape(4096,12288),(12,4096))
        with self.assertRaises(ValueError):M1Backend(0)

    def test_image_generation_requires_digest_and_cuda_smoke(self):
        common={'image_reference':'example/pytorch@sha256:'+'a'*64,'machine':'x86_64',
                'platform':'Linux','python':'3.12','packages':{'torch':'2.8.0','numpy':'2.1.2'}}
        before=dict(common)
        after={**common,'status':'ready_for_gpu_trial','build_smoke':{'exit_code':0},
               'packages':{**common['packages'],'safetensors':'0.8.0'},
               'tools':{'system_packages':{'exit_code':0,'stdout':'g++=13.2.0\nninja-build=1.11.1'}}}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'image'
            self.assertEqual(image.prepare(before,after,path)['status'],'prepared_not_built')
            self.assertIn('safetensors==0.8.0',(path/'addons.lock').read_text())
            self.assertNotIn('torch==',(path/'addons.lock').read_text())
            self.assertIn('FROM example/pytorch@sha256:',(path/'Dockerfile').read_text())
            with self.assertRaises(ValueError):image.prepare(before,{**after,'build_smoke':{}},Path(temp)/'bad')
            with self.assertRaises(ValueError):image.prepare(before,{**after,'image_reference':'example:latest'},Path(temp)/'bad')
            changed={**after,'packages':{**after['packages'],'torch':'2.9.0'}}
            with self.assertRaises(ValueError):image.prepare(before,changed,Path(temp)/'bad')


if __name__=='__main__':unittest.main()
