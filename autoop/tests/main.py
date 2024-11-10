import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline
from autoop.tests.test_artifact import TestArtifact
from autoop.tests.test_elastic_net import TestElasticNet
from autoop.tests.test_knn import TestKNN
from autoop.tests.test_lasso import TestLasso
from autoop.tests.test_metrics import TestAccuracy, TestMAE, TestMCC, \
    TestMSE, TestRMSE, TestSpecificity
from autoop.tests.test_ridge import TestRidge
from autoop.tests.test_sgd import TestSGD
from autoop.tests.test_svc import TestSVC

TestDatabase()
TestStorage()
TestFeatures()
TestPipeline()
TestArtifact()
TestElasticNet()
TestKNN()
TestLasso()
TestAccuracy()
TestMAE()
TestMCC()
TestMSE()
TestRMSE()
TestSpecificity()
TestRidge()
TestSGD()
TestSVC()

if __name__ == '__main__':
    unittest.main()
