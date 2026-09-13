"""Independent checks of the new reference diagnostic, including boundaries."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from aligned_confinement_diagnostic import case, frozen_source


class ConfinementDiagnosticTests(unittest.TestCase):
    def test_frozen_source_matches_dense_system_with_nonzero_endpoints(self):
        h=.1; x=np.arange(-15,16)*h; nu=.4*h; discount=.7
        action=-.6*np.tanh(x); source=np.minimum(np.maximum(np.abs(x)-1,0),.3)
        boundary=.23; n=len(x)-2
        matrix=np.zeros((n,n));rhs=source[1:-1].copy()
        for j,i in enumerate(range(1,len(x)-1)):
            matrix[j,j]=discount+2*nu/h**2
            for k,coefficient in [(i-1,-nu/h**2+action[i]/(2*h)),
                                  (i+1,-nu/h**2-action[i]/(2*h))]:
                if k in (0,len(x)-1):
                    rhs[j]-=coefficient*boundary
                else:
                    matrix[j,k-1]=coefficient
        expected=np.linalg.solve(matrix,rhs)
        value,residual=frozen_source(x,h,nu,discount,action,source,boundary)
        np.testing.assert_allclose(value[1:-1],expected,rtol=0,atol=2e-14)
        self.assertEqual(value[0],boundary);self.assertEqual(value[-1],boundary)
        self.assertLess(residual,2e-14)

    def test_localized_and_unrestricted_values_match_linear_programs(self):
        report,arrays,problem=case(.04,.04,outer=3.)
        x=arrays["x"];h=.04;n=len(x)-2
        for label in ("whole","localized"):
            constraints=[];rhs=[]
            for j,i in enumerate(range(1,len(x)-1)):
                controls=(-1.,1.) if label == "whole" or abs(x[i])<=report["L"]-h else (-np.tanh(x[i]),)
                for action in controls:
                    row=np.zeros(n);row[j]=1+2*problem.nu/h**2
                    if j>0:row[j-1]=-problem.nu/h**2+action/(2*h)
                    if j<n-1:row[j+1]=-problem.nu/h**2-action/(2*h)
                    constraints.append(row)
                    rhs.append(1-action+min(max(abs(x[i])-2,0)**2,.09)/.04)
            result=linprog(-np.ones(n),A_ub=np.asarray(constraints),b_ub=np.asarray(rhs),
                           bounds=[(None,None)]*n,method="highs")
            self.assertTrue(result.success,result.message)
            np.testing.assert_allclose(arrays[label+"_lower"][1:-1],result.x,rtol=0,atol=2e-8)


if __name__ == "__main__":
    unittest.main()
