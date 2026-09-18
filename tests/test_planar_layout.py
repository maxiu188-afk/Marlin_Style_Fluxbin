"""Lossless offline byte planes; independent of CUDA execution."""
import unittest
import torch
from test_deployment import make_payload
from fluxbin_style.deployment import (
    _to_planes, _from_planes, convert_artifact, restore_artifact, decode_layout,
    conversion_record, PLANAR_FORMAT, PLANAR_KERNELS, PackedHybridLinear,
)
from fluxbin_style.engine_interface import M1Backend
from fluxbin_style.factored_reference import structural_reference


class PlanarLayoutTests(unittest.TestCase):
    def test_all_65536_codes_and_each_basis_bit(self):
        word = torch.arange(65536,dtype=torch.int32)
        interleaved = torch.stack((word&255,word>>8),dim=-1).byte()
        planes = _to_planes(interleaved)
        for basis in range(2):
            expected = torch.zeros_like(word)
            for k in range(8):expected |= ((word>>(2*k+basis))&1)<<k
            self.assertTrue(torch.equal(planes[:,basis,0],expected.byte()))
        self.assertTrue(torch.equal(interleaved,_from_planes(planes)))
        self.assertEqual(planes.numel(),interleaved.numel())

    def test_converter_roundtrip_reference_and_storage(self):
        p=make_payload(17,3)
        legacy=convert_artifact(p)
        x=torch.randn(1,384,dtype=torch.bfloat16)
        for kernel in PLANAR_KERNELS:
            layout=convert_artifact(p,kernel=kernel)
            self.assertEqual(layout['codes'].shape,(3,17,2,16))
            self.assertEqual(layout['sparse_codes'].shape,(3,17,2,1))
            for field,value in restore_artifact(layout).items():
                self.assertTrue(torch.equal(value,p[field]),field)
            self.assertTrue(torch.equal(structural_reference(x,legacy),structural_reference(x,layout)))
            self.assertTrue(torch.equal(decode_layout(legacy),decode_layout(layout)))
            rec=conversion_record(p,layout)
            self.assertEqual(rec['format'],PLANAR_FORMAT)
            self.assertEqual(rec['layout_bytes'],conversion_record(p,legacy)['layout_bytes'])
            self.assertEqual(set(layout),set(legacy))

    def test_backend_and_model_adapter_use_planes(self):
        p=make_payload(17,3)
        for kernel in PLANAR_KERNELS:
            backend=M1Backend(kernel=kernel)
            self.assertEqual(backend.contract.format,PLANAR_FORMAT)
            self.assertEqual(backend.contract.arithmetic,'factored_fp32_v1')
            self.assertEqual(backend.convert(p)['codes'].ndim,4)
            module=PackedHybridLinear(p,kernel=kernel,fallback='dense')
            self.assertEqual(module.codes.shape,(3,17,2,16))
            with torch.inference_mode():
                x=torch.randn(2,384,dtype=torch.bfloat16)
                self.assertTrue(torch.equal(module(x),x@decode_layout(convert_artifact(p)).T))
        self.assertEqual(M1Backend(kernel='v4_late').workspace_shape(17,384),(3*272+17,))
