import itertools
import math
import random
import unittest

from text_memory.context_windows import context_windows, same_segment_support, segment_costs
from text_memory.context_completion import complete_context


class ContextWindowTests(unittest.TestCase):
    def test_against_exhaustive_partitions(self):
        rng = random.Random(1526)
        for n in range(1,8):
            vectors = [[rng.uniform(-1,1) for _ in range(3)] for _ in range(n)]
            vectors = [[v / math.sqrt(sum(x*x for x in row)) for v in row] for row in vectors]
            kernel = [[sum(a*b for a,b in zip(x,y)) for y in vectors] for x in vectors]
            for k in range(1,n+1):
                for temperature in [.03,.3,3.]:
                    partitions = []
                    for cuts in itertools.combinations(range(1,n),k-1):
                        edges = [0,*cuts,n]
                        ranges = list(zip(edges,edges[1:]))
                        cost = 0.
                        for a,b in ranges:
                            mean = [sum(row[j] for row in vectors[a:b])/(b-a) for j in range(3)]
                            cost += sum(sum((v-m)**2 for v,m in zip(row,mean)) for row in vectors[a:b])
                        partitions.append((cost,ranges))
                    best = min(c for c,_ in partitions)
                    weights = [math.exp(-(c-best)/temperature) for c,_ in partitions]
                    total = sum(weights)
                    expected = [[0.]*n for _ in range(n)]
                    for weight,(_,ranges) in zip(weights,partitions):
                        for a,b in ranges:
                            for i in range(a,b):
                                for j in range(a,b):
                                    expected[i][j] += weight/total
                    result = same_segment_support(kernel,k,temperature)
                    self.assertAlmostEqual(sum(result['boundary']),k-1,places=8)
                    self.assertAlmostEqual(sum(x['support'] for x in result['intervals']),k,places=8)
                    for i in range(n):
                        for j in range(n):
                            self.assertAlmostEqual(result['same_segment'][i][j],expected[i][j],places=8)

    def test_uncertain_boundary_overlaps_without_cascade(self):
        result = same_segment_support([[1.]*4 for _ in range(4)],2,.1)
        windows = context_windows([[1,2],[3,4]],result['same_segment'],minimum_support=.5)
        self.assertEqual([w['context'] for w in windows],[[1,2],[1,3],[2,4],[3,4]])
        for w in windows:
            self.assertTrue(set(range(w['core'][0],w['core'][1]+1)) <= set(w['fact_numbers']))

    def test_invalid_inputs(self):
        for k,t in [(0,1),(3,1),(1,0),(1,float('nan'))]:
            with self.assertRaises(ValueError):same_segment_support([[1,0],[0,1]],k,t)
        with self.assertRaises(ValueError):segment_costs([[1,2],[3,1]])
        with self.assertRaises(ValueError):context_windows([[1,1]],[[1,0],[0,1]])

    def test_distant_link_is_a_packet_not_a_hull(self):
        # Fact 1 is a distant explanation for fact 4.  The intervening facts
        # must not be pulled into the packet merely because the endpoints link.
        kernel=[[1,.1,.1,.8,.1],[.1,1,.1,.1,.1],[.1,.1,1,.1,.1],[.8,.1,.1,1,.1],[.1,.1,.1,.1,1]]
        support=[[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,0,1,1],[0,0,0,1,1]]
        result=complete_context([[1,3],[4,5]],support,kernel,
            ['A road trip with Mark and Josh','a separate project','another topic',
             'The road trip included a sanctuary','a later unrelated note'], participants=['A','B','B','A','A'],
            minimum_support=.9, link_floor=.7)
        self.assertIn(1,result[3]['fact_numbers'])
        self.assertNotIn(2,result[3]['fact_numbers'])
        self.assertNotIn(3,result[3]['fact_numbers'])


if __name__ == '__main__':
    unittest.main()
