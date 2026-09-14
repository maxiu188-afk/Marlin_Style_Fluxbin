"""CPU index/reference models only; actual PTX is tested in CUDADeploymentTests."""
import unittest
import torch
from test_deployment import make_payload,oracle
from fluxbin_style.deployment import convert_artifact


class MarlinLayoutTests(unittest.TestCase):
    def test_mma_fragment_mapping_reconstructs_combined_weights(self):
        payload=make_payload(19,2);layout=convert_artifact(payload)
        # uint32 words are four consecutive bytes, 16 two-bit signs per word.
        words=layout['codes'].to(torch.int64).reshape(2,19,8,4)
        words=sum(words[:,:,:,i] << (8*i) for i in range(4))
        for dtype in (torch.float16,torch.bfloat16):
            expected=oracle(payload,dtype)
            for g in range(2):
                for tile in (0,16):
                    for step in range(8):
                        reconstructed=torch.zeros(16,16,dtype=dtype);seen=set()
                        for lane in range(32):
                            for reg in range(4):
                                row=lane//4+(reg%2)*8
                                for half in range(2):
                                    col=(lane%4)*2+(reg//2)*8+half
                                    self.assertNotIn((row,col),seen);seen.add((row,col))
                                    o=tile+row;k=step*16+col
                                    if o>=19:continue
                                    code=(int(words[g,o,k//16])>>(2*(k%16)))&3
                                    r=layout['rows'][g,o];c=layout['columns'][g,k]
                                    w=(r[0] if code&1 else -r[0])*c[0]+(r[1] if code&2 else -r[1])*c[1]
                                    selected=int(layout['lookup'][g,k])
                                    if selected>=0:
                                        sp=int(layout['sparse_codes'][g,o,0]) | int(layout['sparse_codes'][g,o,1])<<8
                                        bits=(sp>>(2*selected))&3
                                        r=layout['sparse_rows'][g,o];c=layout['sparse_columns'][g,selected]
                                        w=w+(r[0] if bits&1 else -r[0])*c[0]+(r[1] if bits&2 else -r[1])*c[1]
                                    reconstructed[row,col]=w.to(dtype)
                        self.assertEqual(len(seen),256)
                        count=min(16,19-tile)
                        self.assertTrue(torch.equal(reconstructed[:count],expected[tile:tile+count,g*128+step*16:g*128+step*16+16]))
                        # Replicated B columns compute identical outputs; selected C column 0.
                        x=torch.arange(16,dtype=torch.float32)
                        product=reconstructed.float() @ x[:,None].expand(16,8)
                        outputs={}
                        for lane in range(32):
                            if lane%4==0:
                                outputs[lane//4]=product[lane//4,0]
                                outputs[lane//4+8]=product[lane//4+8,0]
                        torch.testing.assert_close(torch.stack([outputs[i] for i in range(16)]),reconstructed.float()@x)

    def test_padded_code_pitch_broadcast_and_async_alignment(self):
        for word in range(8):
            addresses=[(lane//4)*12+word for lane in range(32)]
            # Same-bank same-address broadcasts are allowed; no distinct-word collisions.
            banks={}
            for address in addresses:banks.setdefault(address%32,set()).add(address)
            self.assertTrue(all(len(words)==1 for words in banks.values()))
        self.assertTrue(all((row*48+half*16)%16==0 for row in range(64) for half in (0,1)))

    def test_three_stage_reuse_and_persistent_task_coverage(self):
        for groups in (1,2,3,4,9,17):
            slots={0:0,1:1}
            for current in range(groups):
                self.assertEqual(slots[current%3],current)
                future=current+2
                self.assertNotEqual(future%3,current%3)
                slots[future%3]=future
        for o,gps in ((1,1),(65,8),(65535,1)):
            groups=33;tiles=(o+63)//64;splits=(groups+gps-1)//gps
            blocks=min(tiles*splits,108*4)
            tasks=[t for b in range(blocks) for t in range(b,tiles*splits,blocks)]
            self.assertEqual(sorted(tasks),list(range(tiles*splits)))
            self.assertEqual(len(tasks),len(set(tasks)))


if __name__=='__main__':unittest.main()
