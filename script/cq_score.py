""" 
This script will calculate the cq score
"""
# Make all variables global for unit testing
DAY_COUNT = 3
TOTAL_DAY_SCORE = 0

first_name_score = 5
last_name_score = 5
dob_score = 5
email_score = 5
phone_no_score = 5
gender_score = 5
orgonization_score = 5
sport_score = 5
adhar_score = 25
pin_code_score = 10
my_people_score = 15
my_place_score = 15
stats_score = {"basic" : 10, "inter" : 10, "advance": 10}
personal_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
edu_doc_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
gold_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
silver_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
bronze_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
participation_cert_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
coach_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
official_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
volunteer_exp_score = {"doc1":10, "doc2":10, "doc3":10, "doc4":10, "doc5":10}
daily_app_visit_score = {"day1":5, "day2":5, "day3":5, "day4":5, "day5":5, "day6":5, "day7":5}

def calculate_cq_score():
    """
    Calculate the CQ score 
    Returns:
        int: The Calculated CQ Score of an athlete
    """
    global DAY_COUNT, TOTAL_DAY_SCORE
    global first_name_score, last_name_score, dob_score, email_score, phone_no_score
    global gender_score, orgonization_score, sport_score, adhar_score, pin_code_score
    global my_people_score, my_place_score, stats_score, personal_doc_score, edu_doc_score
    global gold_cert_score, silver_cert_score, bronze_cert_score, participation_cert_score
    global coach_exp_score, official_exp_score, volunteer_exp_score, daily_app_visit_score

    total_cq_score = 0

    # add the static scores
    total_cq_score = (first_name_score + last_name_score + dob_score + email_score + phone_no_score +
                      gender_score + orgonization_score + sport_score + adhar_score + pin_code_score +
                      my_people_score + my_place_score)

    # add the dynamic scores of documents, certificates and experience
    if len(stats_score) <= 3:
        for stat in stats_score:
            total_cq_score += stats_score[stat]

    if len(personal_doc_score) <= 5:
        for stat in personal_doc_score:
            total_cq_score += personal_doc_score[stat]

    if len(edu_doc_score) <= 5:
        for stat in edu_doc_score:
            total_cq_score += edu_doc_score[stat]

    if len(gold_cert_score) <= 5:
        for stat in gold_cert_score:
            total_cq_score += gold_cert_score[stat]

    if len(silver_cert_score) <= 5:
        for stat in silver_cert_score:
            total_cq_score += silver_cert_score[stat]

    if len(bronze_cert_score) <= 5:
        for stat in bronze_cert_score:
            total_cq_score += bronze_cert_score[stat]

    if len(participation_cert_score) <= 5:
        for stat in participation_cert_score:
            total_cq_score += participation_cert_score[stat]

    if len(coach_exp_score) <= 5:
        for exp in coach_exp_score:
            total_cq_score += coach_exp_score[exp]

    if len(official_exp_score) <= 5:
        for exp in official_exp_score:
            total_cq_score += official_exp_score[exp]

    if len(volunteer_exp_score) <= 5:
        for exp in volunteer_exp_score:
            total_cq_score += volunteer_exp_score[exp]

    # add points if the app visit is regular and reduce the points on absence of app visit
    if DAY_COUNT <= 7 and DAY_COUNT >= 0:
        TOTAL_DAY_SCORE = DAY_COUNT * 5
    elif DAY_COUNT > 7:
        TOTAL_DAY_SCORE = 35
    total_cq_score += TOTAL_DAY_SCORE

    # max cq score is capped at 620, if cq score is exceeding 620 then set it to 620
    if total_cq_score > 620:
        total_cq_score = 620
    return total_cq_score

if __name__ == "__main__":
    print("CQ Score: ", calculate_cq_score())




