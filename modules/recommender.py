"""
modules/recommender.py
Generates personalised recommendations based on student input values.
"""

from typing import List, Tuple


def generate_recommendations(input_dict: dict, dropout_pct: float, placement_pct: float) -> List[Tuple[str, str]]:
    """
    Returns a list of (issue, recommendation) tuples.
    """
    recs = []

    attendance = float(input_dict.get("attendance") or 75)
    cgpa = float(input_dict.get("cgpa") or 7)
    backlogs = int(input_dict.get("backlogs") or 0)
    coding = float(input_dict.get("coding_skill") or 60)
    communication = float(input_dict.get("communication_skill") or 60)
    technical = float(input_dict.get("technical_skill") or 60)
    aptitude = float(input_dict.get("aptitude_score") or 65)
    english = float(input_dict.get("english_proficiency") or 60)
    internship = int(input_dict.get("internship") or 0)
    projects = int(input_dict.get("projects") or 0)
    certifications = int(input_dict.get("certifications") or 0)

    # Academic
    if attendance < 75:
        recs.append(("📉 Low Attendance",
                     "Aim for ≥ 75 % attendance. Regular presence strongly reduces dropout risk."))
    if cgpa < 6.5:
        recs.append(("📚 Low CGPA",
                     "Focus on weak subjects. Seek tutoring and revise fundamentals."))
    if backlogs > 0:
        recs.append(("⚠️ Pending Backlogs",
                     f"Clear your {backlogs} backlog(s) in the next semester — they directly hurt placement odds."))

    # Skills
    if coding < 55:
        recs.append(("💻 Weak Coding Skill",
                     "Practice on LeetCode / HackerRank daily. Start with easy problems for 30 minutes."))
    if communication < 55:
        recs.append(("🗣️ Weak Communication",
                     "Join a public speaking group or debate club. Record yourself and review."))
    if technical < 55:
        recs.append(("🔧 Low Technical Skill",
                     "Build side projects aligned with your branch. Follow YouTube crash courses."))
    if aptitude < 60:
        recs.append(("🧮 Low Aptitude Score",
                     "Practice quantitative aptitude daily using IndiaBix or PrepInsta."))
    if english < 55:
        recs.append(("🌐 English Proficiency",
                     "Read English newspapers, use Duolingo, and write 5 sentences daily."))

    # Experience
    if internship == 0:
        recs.append(("🏢 No Internship",
                     "Apply for a summer internship on Internshala or LinkedIn — even a 1-month stint counts."))
    if projects < 1:
        recs.append(("🛠️ No Projects",
                     "Build at least one end-to-end project and host it on GitHub."))
    if certifications == 0:
        recs.append(("🎓 No Certifications",
                     "Complete a free NPTEL / Coursera / Google certificate relevant to your domain."))

    # Placement-specific
    if placement_pct < 50:
        recs.append(("📝 Low Placement Probability",
                     "Work on your resume, build LinkedIn presence, and do mock interviews."))

    # Dropout-specific
    if dropout_pct > 60:
        recs.append(("🆘 High Dropout Risk",
                     "Talk to your academic advisor immediately. Seek counselling support if needed."))

    if not recs:
        recs.append(("✅ Great Profile!",
                     "Keep up the excellent work. Consider research projects or higher studies."))

    return recs
