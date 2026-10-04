import unittest
from evaluate import evaluate

class EvaluationTests(unittest.TestCase):
    def test_invalid_outputs_remain_errors_and_id_order_is_irrelevant(self):
        gold=[{'id':str(i),'utt':'example','intent':x} for i,x in enumerate(['a','b','b','a'])]
        pred=[{'id':'3','output':'{"intent":"unknown"}'},{'id':'2','output':'not json'},{'id':'1','output':'{"intent":"b"}'},{'id':'0','output':'{"intent":"a"}'}]
        result=evaluate(gold,pred,['a','b'])
        self.assertEqual(result['accuracy'],.5)
        self.assertAlmostEqual(result['macro_f1'],2/3)
        self.assertEqual(result['json_valid_rate'],.75)
        self.assertEqual(result['schema_valid_rate'],.5)
    def test_missing_and_duplicate_predictions_rejected(self):
        gold=[{'id':'1','utt':'x','intent':'a'}]
        for predictions in [[],[{'id':'1','output':'{}'}]*2]:
            with self.assertRaises(ValueError): evaluate(gold,predictions,['a'])

if __name__=='__main__': unittest.main()
