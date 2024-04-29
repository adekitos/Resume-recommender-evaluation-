import spacy
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import fuzz, process
from tabulate import tabulate
import matplotlib.pyplot as plt


def load_skills(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        skills = [skill.strip() for skill in file.readlines()]
    return skills


def extract_skills(text, skills):
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab)
    pattern_skills = [nlp(skill.lower()) for skill in skills]
    matcher.add("SkillMatcher", None, *pattern_skills)
    doc = nlp(text)

    matched_skills = []
    seen_skills = set()  # Keep track of seen skills

    for match_id, start, end in matcher(doc):
        matched_skill = doc[start:end]
        skill_text = matched_skill.text.title()

        # Check if a similar skill has already been seen
        if any(fuzz.ratio(skill_text, seen_skill) >= 90 for seen_skill in seen_skills):
            continue

        matched_skills.append(skill_text)
        seen_skills.add(skill_text)

    # Partial string matching for incomplete skill strings
    for token in doc:
        if token.is_alpha and len(token.text) >= 4:  # Filter tokens with at least 4 characters
            matched_skill, score = process.extractOne(token.text, skills)
            if score >= 90 and matched_skill.title() not in seen_skills:
                matched_skills.append(matched_skill.title())
                seen_skills.add(matched_skill.title())

    return matched_skills


def calculate_skill_matching_score(resume_skills, job_skills):
    true_positives = len(set(resume_skills) & set(job_skills))
    false_positives = len(resume_skills) - true_positives
    false_negatives = len(job_skills) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score


def save_output(file_path, title, data, headers):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"{title}\n")
        file.write(tabulate(data, headers=headers, tablefmt="fancy_grid"))
        file.write("\n\n")


def plot_metrics(precision, recall, f1_score):
    labels = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1_score]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['#FF9F00', '#008B8B', '#8B008B'])
    plt.title('Skill Matching Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score')

    for i, value in enumerate(values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.show()


def main():
    # Load skills from skills.txt
    skills_file = 'devops_engineer_skills.txt'
    all_skills = load_skills(skills_file)

    # Load user's resume
    resume_file = 'devops_engineer_resume.txt'
    with open(resume_file, 'r', encoding='utf-8') as file:
        resume_text = file.read()

    # Extract skills from the user's resume
    user_skills = extract_skills(resume_text, all_skills)
    user_skills = [skill.title() if skill.isupper() or skill.replace(" ", "").isupper() else skill for skill in user_skills]

    # Load job description
    job_description_file = 'devops_engineer_job_post.txt'
    with open(job_description_file, 'r', encoding='utf-8') as file:
        job_description_text = file.read()

    # Extract required skills from the job description
    required_skills = extract_skills(job_description_text, all_skills)
    required_skills = [skill.title() if skill.isupper() or skill.replace(" ", "").isupper() else skill for skill in required_skills]

    # Calculate skill matching score
    precision, recall, f1_score = calculate_skill_matching_score(user_skills, required_skills)

    # Find missing skills (skills in job description but not in resume)
    missing_skills = [skill for skill in required_skills if skill not in user_skills]

    # Save the results to a file
    output_file = 'devops_engineer_output.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('Skills Overview\n')

    save_output(output_file, "All Skills", [[skill] for skill in all_skills], headers=["Skill"])
    save_output(output_file, "User's Skills", [[skill] for skill in user_skills], headers=["Skill"])
    save_output(output_file, "Relevant Skills", [[skill] for skill in required_skills], headers=["Skill"])
    save_output(output_file, "Recommended Skills (Not in Resume)", [[skill] for skill in missing_skills], headers=["Skill"])

    # Plot the skill matching evaluation metrics
    plot_metrics(precision, recall, f1_score)


if __name__ == '__main__':
    main()