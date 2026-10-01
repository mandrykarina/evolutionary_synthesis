import unittest
import numpy as np
from model import generate,repair,feasible,evaluate,fronts,crowding,survive,solve,coverage

class Tests(unittest.TestCase):
    def setUp(self):self.data=generate()
    def test_fronts_hand_calculated(self):
        f=fronts(np.array([[0,2],[1,1],[2,0],[2,2],[3,3]]))
        self.assertEqual(set(f[0]),{0,1,2});self.assertEqual(list(f[1]),[3]);self.assertEqual(list(f[2]),[4])
    def test_equal_points(self):self.assertEqual(len(fronts(np.ones((4,3)))[0]),4)
    def test_crowding(self):
        d=crowding(np.array([[0,2],[1,1],[2,0]]));self.assertTrue(np.isinf(d[0]) and np.isinf(d[2]));self.assertEqual(d[1],2)
        self.assertFalse(np.isnan(crowding(np.ones((4,3)))).any())
    def test_repair(self):
        rng=np.random.default_rng(1)
        for x in rng.integers(-5,40,(100,12)):self.assertTrue(feasible(repair(x,self.data),self.data))
    def test_objective_hand_calculated(self):
        d=dict(items=['x'],prices=[2],volumes=[3],importance=[4],demand=[[1],[3]],max_quantity=20,warehouse_capacity=340,budget=3200)
        np.testing.assert_allclose(evaluate([[2]],d),[[4,4,1.5]])
    def test_coverage(self):self.assertEqual(coverage(np.array([[0,0]]),np.array([[1,1],[2,3]])),1)
    def test_budget_and_determinism(self):
        c=dict(population=8,generations=3,mutation_probability=.1,crossover_probability=.9)
        a=solve(c,self.data,17);b=solve(c,self.data,17)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[3],32);self.assertEqual(len(fronts(a[1])[0]),len(a[1]));self.assertTrue(all(feasible(x,self.data) for x in a[0]))
if __name__=='__main__':unittest.main()
