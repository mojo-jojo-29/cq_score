import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../script')))
import cq_score
import unittest


class TestCqScore(unittest.TestCase):

    def test_TC01(self):
        """
        Day count streak is 3 and all other parameters are set to maximum values.
        """
        cq_score.DAY_COUNT = 3
        cq_score.TOTAL_DAY_SCORE = 0
        cq_score.first_name_score = 5
        cq_score.last_name_score = 5
        cq_score.dob_score = 5
        cq_score.email_score = 5
        cq_score.phone_no_score = 5
        cq_score.gender_score = 5
        cq_score.orgonization_score = 5
        cq_score.sport_score = 5
        cq_score.adhar_score = 25
        cq_score.pin_code_score = 10
        cq_score.my_people_score = 15
        cq_score.my_place_score = 15
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 600, msg="Test Case TC01 executed")
        
    def test_TC02(self):
        """
        Day count is greater than 7 and other values are set at maximum
        """
        cq_score.DAY_COUNT = 8
        cq_score.TOTAL_DAY_SCORE = 0
        cq_score.first_name_score = 5
        cq_score.last_name_score = 5
        cq_score.dob_score = 5
        cq_score.email_score = 5
        cq_score.phone_no_score = 5
        cq_score.gender_score = 5
        cq_score.orgonization_score = 5
        cq_score.sport_score = 5
        cq_score.adhar_score = 25
        cq_score.pin_code_score = 10
        cq_score.my_people_score = 15
        cq_score.my_place_score = 15
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 620, msg="Test Case TC02 executed")
        

    def test_TC02(self):
        """
        Day count is negative and other values are set at maximum
        """
        cq_score.DAY_COUNT = -8
        cq_score.TOTAL_DAY_SCORE = 0
        cq_score.first_name_score = 5
        cq_score.last_name_score = 5
        cq_score.dob_score = 5
        cq_score.email_score = 5
        cq_score.phone_no_score = 5
        cq_score.gender_score = 5
        cq_score.orgonization_score = 5
        cq_score.sport_score = 5
        cq_score.adhar_score = 25
        cq_score.pin_code_score = 10
        cq_score.my_people_score = 15
        cq_score.my_place_score = 15
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 585, msg="Test Case TC03 executed")

    def test_TC04(self):
        """
        Day count = 0 and other values are set at maximum
        """
        cq_score.DAY_COUNT = 0
        cq_score.TOTAL_DAY_SCORE = 0
        cq_score.first_name_score = 5
        cq_score.last_name_score = 5
        cq_score.dob_score = 5
        cq_score.email_score = 5
        cq_score.phone_no_score = 5
        cq_score.gender_score = 5
        cq_score.orgonization_score = 5
        cq_score.sport_score = 5
        cq_score.adhar_score = 25
        cq_score.pin_code_score = 10
        cq_score.my_people_score = 15
        cq_score.my_place_score = 15
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 585, msg="Test Case TC04 executed")


    def test_TC05(self):
        """
        Day count = 3 and scaler values are set at 0 and others to max
        """
        cq_score.DAY_COUNT = 3
        cq_score.TOTAL_DAY_SCORE = 0
        cq_score.first_name_score = 0
        cq_score.last_name_score = 0
        cq_score.dob_score = 0
        cq_score.email_score = 0
        cq_score.phone_no_score = 0
        cq_score.gender_score = 0
        cq_score.orgonization_score = 0
        cq_score.sport_score = 0
        cq_score.adhar_score = 0
        cq_score.pin_code_score = 0
        cq_score.my_people_score = 0
        cq_score.my_place_score = 0
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 495, msg="Test Case TC05 executed")

    
    def test_TC06(self):
        """
        Day count = 3 and scaler values are set at max and others changed to few zeros
        """
        cq_score.DAY_COUNT = 3
        cq_score.TOTAL_DAY_SCORE = 5
        cq_score.first_name_score = 5
        cq_score.last_name_score = 5
        cq_score.dob_score = 5
        cq_score.email_score = 5
        cq_score.phone_no_score = 5
        cq_score.gender_score = 5
        cq_score.orgonization_score = 5
        cq_score.sport_score = 5
        cq_score.adhar_score = 25
        cq_score.pin_code_score = 10
        cq_score.my_people_score = 15
        cq_score.my_place_score = 15
        cq_score.stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
        cq_score.personal_doc_score = {"doc1":10, "doc2":10, "doc3":10}
        cq_score.edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10}
        cq_score.gold_cert_score = {"doc1":10, "doc2":10}
        cq_score.silver_cert_score = {"doc1":10}
        cq_score.bronze_cert_score = {}
        cq_score.participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
        cq_score.coach_exp_score = {"doc1":10, "doc2":10, "doc3":10}
        cq_score.official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10}
        cq_score.volunteer_exp_score = {"doc1":10, "doc2":10}
        cq_score.daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

        self.assertEqual(cq_score.calculate_cq_score(), 390, msg="Test Case TC06 executed")

if __name__ == "__main__":
    unittest.main()