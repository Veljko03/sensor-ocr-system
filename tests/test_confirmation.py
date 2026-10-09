import hashlib
import concurrent.futures
import unittest
from frame_confirmation import FrameConfirmation


def fingerprint(name):return hashlib.sha256(name.encode()).hexdigest()


class ConfirmationTests(unittest.TestCase):
    def observe(self, machine, value, sequence, time, device='esp_1', name=None, now=None):
        return machine.observe(device,value,fingerprint(name or str(sequence)),sequence,time,time if now is None else now)

    def test_zero_requires_two_distinct_frames(self):
        m=FrameConfirmation();self.assertIsNone(self.observe(m,0.,1,0.).value)
        decision=self.observe(m,0.,2,5.)
        self.assertTrue(decision.confirmed);self.assertEqual(decision.value,0.)

    def test_rejected_frame_resets_chain(self):
        m=FrameConfirmation();self.observe(m,6.62,1,0.)
        self.observe(m,None,2,5.)
        self.assertFalse(self.observe(m,6.62,3,10.).confirmed)

    def test_changed_value_resets_chain(self):
        m=FrameConfirmation();self.observe(m,6.62,1,0.)
        self.assertFalse(self.observe(m,6.63,2,5.).confirmed)
        self.assertTrue(self.observe(m,6.63,3,10.).confirmed)

    def test_duplicate_does_not_confirm_and_resets(self):
        m=FrameConfirmation();self.observe(m,6.62,1,0.,name='A')
        self.assertEqual(self.observe(m,6.62,2,5.,name='A').reason,'duplicate_frame')
        self.assertFalse(self.observe(m,6.62,3,10.,name='B').confirmed)

    def test_replay_after_intermediate_frame_rejected(self):
        m=FrameConfirmation();self.observe(m,6.62,1,0.,name='A');self.observe(m,6.62,2,5.,name='B')
        self.assertEqual(self.observe(m,6.62,3,10.,name='A').reason,'duplicate_frame')

    def test_interleaved_devices_have_separate_state(self):
        m=FrameConfirmation(devices=('esp_1','verified_test_device'))
        self.observe(m,6.62,1,0.);self.observe(m,6.62,1,1.,device='verified_test_device')
        self.assertTrue(self.observe(m,6.62,2,5.).confirmed)
        self.assertFalse(self.observe(m,None,2,6.,device='verified_test_device').confirmed)

    def test_unsupported_device_cannot_use_festo_assumptions(self):
        self.assertEqual(self.observe(FrameConfirmation(),0.,1,0.,device='esp_2').reason,'unsupported_device')

    def test_processing_delay_future_time_and_gap(self):
        m=FrameConfirmation();self.assertEqual(self.observe(m,6.62,1,0.,now=11.).reason,'stale_processing')
        self.assertEqual(self.observe(m,6.62,2,20.,now=19.).reason,'stale_processing')
        self.observe(m,6.62,3,25.)
        self.assertFalse(self.observe(m,6.62,4,50.).confirmed)

    def test_concurrent_duplicate_ingress_cannot_double_confirm(self):
        m=FrameConfirmation()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _:self.observe(m,6.62,1,0.),range(16)))
        self.assertFalse(any(r.confirmed for r in results))
        self.assertEqual(sum(r.reason=='awaiting_confirmation' for r in results),1)

    def test_old_completion_does_not_replace_newer_state(self):
        m=FrameConfirmation();self.observe(m,6.62,2,5.)
        self.assertEqual(self.observe(m,1.61,1,0.,now=6.).reason,'out_of_order')
        self.assertTrue(self.observe(m,6.62,3,10.).confirmed)

    def test_burst_and_missing_sequence_reset(self):
        m=FrameConfirmation();self.observe(m,6.62,1,0.)
        self.assertEqual(self.observe(m,6.62,2,.2).reason,'frames_too_close')
        self.observe(m,6.62,3,5.)
        self.assertFalse(self.observe(m,6.62,5,10.).confirmed)

    def test_restart_does_not_reuse_chain(self):
        self.observe(FrameConfirmation(),6.62,1,0.)
        self.assertFalse(self.observe(FrameConfirmation(),6.62,2,5.).confirmed)

    def test_invalid_values_fail_closed(self):
        for value in (float('nan'),float('inf'),-.01,10.,.001,True,'6.62'):
            with self.subTest(value=value):self.assertEqual(self.observe(FrameConfirmation(),value,1,0.).reason,'invalid_value')


if __name__=='__main__':unittest.main()
