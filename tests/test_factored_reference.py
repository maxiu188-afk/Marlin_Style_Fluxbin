import unittest
import torch
from test_deployment import make_payload, oracle
from fluxbin_style.deployment import convert_artifact, workspace_shape
from fluxbin_style.factored_reference import structural_reference, structural_gate
from fluxbin_style.packing import unpack_two_bases


class FactoredReferenceTests(unittest.TestCase):
    def test_fp64_reconstruction_matches_independent_factored_dot(self):
        for o,g in ((1,1),(17,3)):
            p=make_payload(o,g);layout=convert_artifact(p)
            x=torch.randn(1,g*128).bfloat16();xd=x.double().reshape(g,128)
            signs=unpack_two_bases(p['global_sign_codes']).double().reshape(2,o,g,128)
            z=xd[None,:,:]*p['global_column_scales'].double()
            expected=((signs*z[:,None,:,:]).sum(-1)*p['global_row_scales'].double()).sum((0,2))
            sparse=unpack_two_bases(p['refinement_sign_codes']).double().reshape(2,o,g,8)
            sx=xd.gather(1,p['refinement_indices'].long())
            sz=sx[None,:,:]*p['refinement_column_scales'].double()
            expected+=((sparse*sz[:,None,:,:]).sum(-1)*p['refinement_row_scales'].double()).sum((0,2))
            actual=structural_reference(x,layout)
            torch.testing.assert_close(actual,expected[None,:],atol=1e-12,rtol=1e-12)
            self.assertTrue(structural_gate(actual.bfloat16(),actual)['passed'])
            legacy=x.double()@oracle(p,torch.bfloat16).double().t()
            self.assertGreater((actual-legacy).abs().max().item(),1e-5)

    def test_nonfinite_and_shape_fail_closed(self):
        self.assertFalse(structural_gate(torch.tensor([[float('nan')]]).bfloat16(),torch.ones(1,1).double())['passed'])
        with self.assertRaises(ValueError):structural_reference(torch.zeros(1,1),convert_artifact(make_payload()))

    def test_workspace_and_transform_permutation(self):
        self.assertEqual(workspace_shape(17,384,4,kernel='v4'),(3*272+17,))
        self.assertEqual(workspace_shape(17,384,4),(1,17))
        perm=[(k%4)*32+k//4 for k in range(128)]
        self.assertEqual(sorted(perm),list(range(128)))
        for j in range(4):
            self.assertEqual([perm[lane*4+j] for lane in range(32)],list(range(j*32,(j+1)*32)))


if __name__=='__main__':unittest.main()
