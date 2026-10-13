"""Watch strict-past reconstruction from physical events, not like prefixes."""
import importlib.util
import unittest

class WatchTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_watch'),'watch reconstruction behavior missing')
        from vkgr import retrieval_watch
        return retrieval_watch

    def fixture(self):
        events=[{'user_id':1,'item_id':0,'like':True},{'user_id':1,'item_id':2,'like':False},{'user_id':1,'item_id':2,'like':False},{'user_id':1,'item_id':3,'like':True},{'user_id':1,'item_id':4,'like':False}]
        example={'example_id':0,'user_id':1,'history':[0],'history_positions':[[0,0]],'target':3,'target_position':[0,3],'past_likes':[0]}
        query={'user_id':1,'history':[0,3],'history_positions':[[0,0],[0,3]],'target':99,'target_position':[25,0],'oov':True}
        return events,example,query

    def test_watch_preserves_duplicate_and_excludes_target_but_eval_includes_all_train(self):
        m=self.module();events,e,q=self.fixture()
        train,validation,slices=m.build_watch([(0,events)],[e],[q],[0,2,3,4])
        self.assertEqual(train,[{'example_id':0,'history':[0,2,2],'history_positions':[[0,0],[0,1],[0,2]]}])
        self.assertEqual(validation[0]['history'],[0,2,2,3,4])
        self.assertEqual({k:v for k,v in validation[0].items() if k not in ('history','history_positions')},{k:v for k,v in q.items() if k not in ('history','history_positions')})
        self.assertEqual(slices['user_like_counts'],{'1':2})
        self.assertEqual(e['past_likes'],[0])

    def test_watch_rejects_target_position_like_and_unknown_catalog_drift(self):
        m=self.module();events,e,q=self.fixture()
        for changed in ({**e,'target_position':[0,2]}, {**e,'past_likes':[0,2]}):
            with self.assertRaises(ValueError):m.build_watch([(0,events)],[changed],[q],[0,2,3,4])
        with self.assertRaises(ValueError):m.build_watch([(0,events)],[e],[q],[0,2,3])

    def test_last50_counts_observed_events_and_train_defined_thresholds(self):
        m=self.module()
        events=[{'user_id':1,'item_id':0,'like':True}]+[{'user_id':1,'item_id':2,'like':False} for _ in range(55)]+[{'user_id':1,'item_id':3,'like':True}]
        e={'example_id':0,'user_id':1,'history':[0],'history_positions':[[0,0]],'target':3,'target_position':[0,56],'past_likes':[0]}
        q={'user_id':1,'history':[0,3],'history_positions':[[0,0],[0,56]],'target':3,'target_position':[25,0],'oov':False}
        train,val,slices=m.build_watch([(0,events)],[e],[q],[0,2,3])
        self.assertEqual(train[0]['history'],[2]*50)
        self.assertEqual(train[0]['history_positions'][0],[0,6])
        self.assertEqual(val[0]['history'],[2]*49+[3])
        self.assertEqual(slices['popularity_thresholds'],{'median':1,'p90':1})
