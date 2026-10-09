import unittest
import cv2
import numpy as np
from ocr_candidate import decode_glyph,read_real_meter_image


class OcrCandidateTests(unittest.TestCase):
    def test_real_zero_shape_and_narrow_one(self):
        zero=np.zeros((120,80),np.uint8)
        cv2.rectangle(zero,(0,0),(79,119),255,-1)
        cv2.rectangle(zero,(14,18),(65,101),0,-1)
        self.assertEqual(decode_glyph(zero)[0],'0')
        one=np.full((120,12),255,np.uint8)
        self.assertEqual(decode_glyph(one)[0],'1')

    def test_solid_and_horizontal_stripes_are_not_digits(self):
        filled=np.full((120,80),255,np.uint8)
        striped=np.zeros((120,80),np.uint8)
        for y in range(0,120,12):striped[y:y+6]=255
        self.assertIsNone(decode_glyph(filled)[0]);self.assertIsNone(decode_glyph(striped)[0])

    def test_blue_object_and_ceiling_like_stripes_have_no_measurement(self):
        blank=np.full((240,320,3),(120,50,20),np.uint8)
        self.assertIsNone(read_real_meter_image(blank)[0])
        scene=np.zeros((240,320,3),np.uint8)
        cv2.rectangle(scene,(90,50),(210,170),(130,60,25),-1)
        for y in range(55,170,15):cv2.rectangle(scene,(95,y),(205,y+6),(230,230,220),-1)
        self.assertIsNone(read_real_meter_image(scene)[0])

    def test_empty_frame_fails_closed(self):
        for image in (None,np.zeros((0,0,3),np.uint8)):
            self.assertIsNone(read_real_meter_image(image)[0])


if __name__=='__main__':unittest.main()
