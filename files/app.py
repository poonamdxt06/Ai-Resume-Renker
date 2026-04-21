import os
import json
import re
import pdfplumber
from flask import Flask, render_template, request, jsonify, session, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import tempfile


app = Flask(__name__)
app.secret_key = "resume_analyzer_secret_2024"

# ─────────────────────────────────────────────
# DATA DEFINITIONS
# ─────────────────────────────────────────────

SKILLS_DB = {
    "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
                    "rust", "php", "swift", "kotlin", "scala", "r", "matlab", "bash", "shell"],
    "web": ["html", "css", "react", "angular", "vue", "nodejs", "express", "django", "flask",
            "fastapi", "spring", "bootstrap", "tailwind", "jquery", "graphql", "rest", "api"],
    "data": ["sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis", "cassandra",
             "elasticsearch", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly"],
    "ml_ai": ["machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
              "keras", "scikit-learn", "xgboost", "lightgbm", "transformers", "bert", "gpt",
              "neural network", "random forest", "regression", "classification", "clustering"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd",
              "git", "github", "gitlab", "linux", "devops", "microservices"],
    "bi_analytics": ["excel", "power bi", "tableau", "looker", "qlik", "dax", "etl",
                     "data warehouse", "spark", "hadoop", "airflow", "dbt"],
    "soft": ["communication", "teamwork", "leadership", "problem solving", "agile", "scrum",
             "project management", "jira", "confluence"]
}

ALL_SKILLS = [skill for group in SKILLS_DB.values() for skill in group]

ROLE_KEYWORDS = {

    # 📊 DATA ROLES
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "dashboard", "kpi", "reporting", "data cleaning", "visualization"],
    "Data Scientist": ["machine learning", "statistics", "python", "r", "modeling", "analysis", "prediction", "nlp", "data science"],
    "Data Engineer": ["etl", "data pipeline", "spark", "hadoop", "airflow", "big data", "warehouse", "dbt"],

    # 🤖 AI / ML
    "Machine Learning Engineer": ["machine learning", "tensorflow", "pytorch", "model", "training", "algorithm", "prediction"],
    "AI Engineer": ["ai", "deep learning", "neural network", "nlp", "computer vision", "transformers"],

    # 💻 DEVELOPMENT
    "Frontend Developer": ["html", "css", "javascript", "react", "angular", "vue", "frontend", "ui"],
    "Backend Developer": ["nodejs", "python", "java", "api", "rest", "django", "spring", "database"],
    "Full Stack Developer": ["frontend", "backend", "react", "nodejs", "api", "full stack"],
    "Software Developer": ["programming", "software", "development", "oop", "debugging", "git"],
    "Mobile App Developer": ["android", "ios", "flutter", "react native", "kotlin", "swift"],

    # ☁️ CLOUD / DEVOPS
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "jenkins", "aws", "azure", "gcp", "devops"],
    "Cloud Engineer": ["aws", "azure", "gcp", "cloud", "infrastructure", "deployment"],

    # 🔐 SECURITY
    "Cyber Security Analyst": ["security", "penetration testing", "vulnerability", "network security", "ethical hacking"],

    # 🧪 TESTING
    "QA Engineer": ["testing", "manual testing", "automation testing", "selenium", "test cases", "qa"],

    # 🗄️ DATABASE
    "Database Administrator": ["sql", "oracle", "mysql", "database", "backup", "performance tuning"],

    # 🎨 DESIGN
    "UI/UX Designer": ["figma", "adobe xd", "wireframe", "prototype", "ui", "ux", "design"],

    # 📈 BUSINESS / NON-TECH
    "Business Analyst": ["business analysis", "requirement", "stakeholder", "documentation", "process", "uml", "use case", "gap analysis", "business process modeling"],
    "Product Manager": ["product", "roadmap", "strategy", "user story", "agile", "scrum", "jira", "confluence", "stakeholder", "market research", "competitive analysis"],
    "Project Manager": ["project management", "planning", "execution", "scrum", "jira", "confluence", "stakeholder", "risk management", "budgeting", "resource allocation", "project lifecycle", "project scheduling", "project coordination", "project communication", "project documentation", "project leadership", "project delivery"],
    "HR": ["recruitment", "talent", "hr", "interview", "employee", "onboarding", "performance review", "hrms", "payroll", "employee relations", "compensation", "benefits", "training", "development", "hr analytics", "hr strategy", "compliance", "labor laws", "employee engagement", "diversity and inclusion", "workforce planning", "succession planning", "hr technology"],
    "Market Research": ["marketing", "seo", "campaign", "social media", "branding", "market research", "analytics", "content marketing", "email marketing", "ppc", "google analytics", "customer segmentation", "competitive analysis", "marketing strategy", "influencer marketing", "affiliate marketing", "conversion rate optimization", "marketing automation", "copywriting", "public relations", "event marketing", "product marketing", "digital marketing", "growth hacking", "marketing analytics", "customer journey", "brand management", "market trends", "consumer behavior"],
    "Sales": ["sales", "client", "revenue", "lead generation", "crm", "negotiation", "closing", "salesforce", "account management", "sales strategy", "sales operations", "sales analytics", "customer relationship management", "sales pipeline", "sales forecasting", "sales enablement", "sales training", "sales performance", "sales targets", "sales presentations"],
    "Content Writer": ["content", "writing", "blog", "seo writing", "copywriting", "editor", "proofreading", "content strategy", "social media content", "technical writing", "creative writing", "content marketing", "ghostwriting", "scriptwriting", "email copywriting", "product description writing", "white paper writing", "case study writing", "press release writing"],
    "Customer Support": ["support", "customer", "service", "helpdesk", "troubleshooting", "ticketing", "customer satisfaction", "crm", "customer retention", "customer feedback", "customer communication", "customer relationship management", "customer service skills", "customer support tools", "customer support strategy", "customer support analytics", "customer support training"],

    # 🧾 GENERAL
    "General Professional": []
}


YOUTUBE_RESOURCES = {
    "python": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
    "sql": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
    "machine learning": "https://www.youtube.com/watch?v=GwIo3gDZCVQ",
    "deep learning": "https://www.youtube.com/watch?v=aircAruvnKk",
    "tensorflow": "https://www.youtube.com/watch?v=tPYj3fFJGjk",
    "pytorch": "https://www.youtube.com/watch?v=Z_ikDlimN6A",
    "react": "https://www.youtube.com/watch?v=SqcY0GlETPk",
    "nodejs": "https://www.youtube.com/watch?v=Oe421EPjeBE",
    "docker": "https://www.youtube.com/watch?v=fqMOX6JJhGo",
    "kubernetes": "https://www.youtube.com/watch?v=X48VuDVv0do",
    "aws": "https://www.youtube.com/watch?v=ulprqHHWlng",
    "excel": "https://www.youtube.com/watch?v=Vl0H-qTclOg",
    "power bi": "https://www.youtube.com/watch?v=TmhQCQr_DXXI",
    "tableau": "https://www.youtube.com/watch?v=TPMlZxRRaBQ",
    "javascript": "https://www.youtube.com/watch?v=PkZNo7MFNFg",
    "java": "https://www.youtube.com/watch?v=eIrMbAQSU34",
    "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "nlp": "https://www.youtube.com/watch?v=X2vAabgKiuM",
    "pandas": "https://www.youtube.com/watch?v=vmEHCJofslg",
    "scikit-learn": "https://www.youtube.com/watch?v=0B5eIE_1vpU",
    "default": "https://www.youtube.com/results?search_query="
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text = ""
    return text.strip()


def extract_skills(text):
    """Extract skills present in text."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return list(set(found))


def get_missing_skills(resume_skills, jd_skills):
    """Return skills in JD but missing from resume."""
    r = set(s.lower() for s in resume_skills)
    j = set(s.lower() for s in jd_skills)
    return [s.title() for s in (j - r)]


def compute_ats_score(resume_text, jd_text):
    """Compute cosine similarity TF-IDF score."""
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform([resume_text, jd_text])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return 0.0


def extract_top_keywords(text, n=15):
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=100,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vec.fit_transform([text])
        feature_names = vec.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        word_scores = list(zip(feature_names, scores))

        filtered = []
        for word, score in word_scores:
            if (
                len(word) > 3 and
                score > 0.1 and
                not re.search(r'\d', word) and   # ❌ remove numbers
                word not in ["email", "phone", "number"]
            ):
                filtered.append((word, score))

        sorted_words = sorted(filtered, key=lambda x: x[1], reverse=True)

        return [word for word, score in sorted_words[:n]]

    except Exception:
        return []




def detect_role(text):
    text = text.lower()
    role_scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        if not keywords:
            continue

        match_count = 0

        for kw in keywords:
            if kw in text:
                match_count += 1

        score = match_count / len(keywords)
        role_scores[role] = score

    best_role = max(role_scores, key=role_scores.get)
    best_score = role_scores[best_role]

    if best_score < 4 / 7:  # Require at least ~57% keyword match for a confident role detection
        return "General Professional", best_score * 100 

    return best_role, round(best_score * 100, 1)


def detect_jd_role(text):
    text = text.lower()

    # direct title match
    for role in ROLE_KEYWORDS.keys():
        if role.lower() in text:
            return role, 100

    # fallback
    return detect_role(text)

    # 🔍 Keyword based fallback
    role_scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        role_scores[role] = score

    best_role = max(role_scores, key=role_scores.get)
    best_score = role_scores[best_role]

    if best_score == 0:
        return "General Professional", 0

    return best_role, best_score * 10





def generate_suggestions(missing_skills, missing_keywords, jd_role, resume_role, resume_skills, ats_score):
    """Generate actionable AI suggestions."""
    suggestions = []

    if missing_skills:
        top_missing = missing_skills[:3]
        suggestions.append(f"🎯 Add hands-on projects showcasing: {', '.join(top_missing)} to strengthen your profile.")

    if missing_keywords:
        top_kw = missing_keywords[:4]
        suggestions.append(f"📝 Incorporate these keywords naturally in your experience section: {', '.join(top_kw)}.")

    if missing_skills:
        suggestions.append(f"🎯 Focus on learning: {', '.join(missing_skills[:3])}")

    if jd_role == "Data Analyst":
        suggestions.append("📈 Highlight any dashboards or reports you built — mention the tool (Power BI/Tableau) and business impact.")
    elif jd_role == "Machine Learning Engineer":
        suggestions.append("🤖 List model accuracy metrics and dataset sizes in project descriptions.")
    elif jd_role in ["Web Developer", "Backend Developer"]:
        suggestions.append("💻 Link live demos or GitHub repos directly in your resume for each major project.")

    return suggestions

def generate_dynamic_resume(resume_text, jd_text, jd_role, jd_skills, resume_skills, missing_skills, jd_keywords):
    """Generate a dynamically customized resume based on job description."""
    
    # Extract sections from original resume
    resume_lower = resume_text.lower()
    
    # Build role-specific templates
    role_templates = {
        "Data Analyst": {
            "summary": f"Data Analyst with expertise in {', '.join(jd_skills[:4] if jd_skills else ['data analysis', 'visualization'])}. Proven ability to transform raw data into actionable insights and drive business decisions.",
            "focus_skills": ["SQL", "Excel", "Power BI", "Tableau", "Data Analysis", "Visualization"],
            "achievement_keywords": ["dashboard", "kpi", "reporting", "analysis", "insight", "metric"]
        },
        "Data Scientist": {
            "summary": f"Data Scientist proficient in {', '.join(jd_skills[:4] if jd_skills else ['machine learning', 'python', 'statistics'])}. Expertise in building predictive models and driving data-driven solutions.",
            "focus_skills": ["Python", "Machine Learning", "Statistics", "Modeling", "SQL"],
            "achievement_keywords": ["model", "accuracy", "prediction", "algorithm", "optimization"]
        },
        "Machine Learning Engineer": {
            "summary": f"Machine Learning Engineer specializing in {', '.join(jd_skills[:4] if jd_skills else ['deep learning', 'tensorflow', 'pytorch'])}. Experience building and deploying ML models at scale.",
            "focus_skills": ["TensorFlow", "PyTorch", "Deep Learning", "Model Training", "Optimization"],
            "achievement_keywords": ["model accuracy", "neural network", "training", "optimization", "dataset"]
        },
        "Backend Developer": {
            "summary": f"Backend Developer with strong experience in {', '.join(jd_skills[:4] if jd_skills else ['nodejs', 'python', 'api design'])}. Skilled in building scalable, reliable APIs and services.",
            "focus_skills": ["Node.js", "Python", "API Design", "Database", "REST"],
            "achievement_keywords": ["api", "endpoint", "service", "performance", "deployment"]
        },
        "Frontend Developer": {
            "summary": f"Frontend Developer experienced in {', '.join(jd_skills[:4] if jd_skills else ['react', 'javascript', 'ui design'])}. Passionate about creating responsive and user-friendly interfaces.",
            "focus_skills": ["React", "JavaScript", "CSS", "HTML", "UI/UX"],
            "achievement_keywords": ["component", "responsive", "ui", "performance", "optimization"]
        },
        "DevOps Engineer": {
            "summary": f"DevOps Engineer with expertise in {', '.join(jd_skills[:4] if jd_skills else ['docker', 'kubernetes', 'ci/cd'])}. Experienced in infrastructure automation and deployment pipelines.",
            "focus_skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Jenkins"],
            "achievement_keywords": ["deployment", "automation", "infrastructure", "pipeline", "scaling"]
        },
        "General Professional": {
            "summary": f"Professional with expertise in {', '.join(jd_skills[:4] if jd_skills else ['problem solving', 'leadership'])}. Strong track record of delivering results and driving impact.",
            "focus_skills": list(set(jd_skills))[:6] if jd_skills else [],
            "achievement_keywords": ["achievement", "result", "impact", "delivered", "improved"]
        }
    }
    
    # Get template for detected role
    template = role_templates.get(jd_role, role_templates["General Professional"])
    
    # Build improved resume
    improved_resume = f"""PROFESSIONAL SUMMARY
{template['summary']}

CORE SKILLS
{', '.join(template['focus_skills'] if template['focus_skills'] else jd_skills[:8])}

TECHNICAL SKILLS
{', '.join(jd_skills if jd_skills else ['Analysis', 'Problem Solving', 'Communication'])}

KEY ACHIEVEMENTS & EXPERIENCE
• Demonstrated proficiency in {', '.join(jd_skills[:3] if jd_skills else ['core technologies'])}
• Delivered solutions that improved business metrics and efficiency
• Collaborated with cross-functional teams to achieve project goals
• Continuously learned and adopted new {jd_role.lower()} best practices and tools

PROJECTS & PORTFOLIO
• [{jd_role} Project 1] - Implemented solution using {jd_skills[0] if jd_skills else 'key technologies'}
  - Achieved measurable results and business impact
  - Tech Stack: {', '.join(jd_skills[:3] if jd_skills else ['relevant tools'])}

• [{jd_role} Project 2] - Led end-to-end development/analysis project
  - Utilized {', '.join(jd_skills[3:5] if len(jd_skills) > 3 else jd_skills)} for implementation
  - Delivered on time with high quality standards

PROFESSIONAL DEVELOPMENT
• Proficient in: {', '.join(jd_skills[:5] if jd_skills else ['core competencies'])}
• Actively developing skills in: {', '.join(missing_skills[:3] if missing_skills else ['emerging technologies'])}
• Strong foundation in {jd_role.lower()} principles and best practices

EDUCATION
[Add Your Degree] - [University Name]
Relevant Coursework: {', '.join(jd_keywords[:5] if jd_keywords else ['relevant topics'])}

CERTIFICATIONS
[Add relevant certifications related to {jd_role}]
"""
    
    return improved_resume.strip()

def get_resource_links(missing_skills):
    """Get YouTube learning resource links for missing skills."""
    links = []
    for skill in missing_skills[:6]:
        skill_lower = skill.lower()
        url = YOUTUBE_RESOURCES.get(skill_lower,
              YOUTUBE_RESOURCES["default"] + skill_lower.replace(" ", "+"))
        links.append({"skill": skill, "url": url})
    return links


def compute_breakdown(resume_text, jd_text, resume_skills, jd_skills, resume_keywords, jd_keywords):
    """Compute ATS breakdown scores."""
    # Skills score
    if jd_skills:
        matched = len(set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills))
        skills_score = round((matched / len(jd_skills)) * 100, 1)
    else:
        skills_score = 50.0

    # Keyword score
    if jd_keywords:
        resume_lower = resume_text.lower()
        matched_kw = sum(1 for kw in jd_keywords if kw.lower() in resume_lower)
        keyword_score = round((matched_kw / len(jd_keywords)) * 100, 1)
    else:
        keyword_score = 50.0

    # Content score (based on resume length and structure indicators)
    content_indicators = ["experience", "education", "skills", "project", "achievement",
                          "certification", "summary", "objective"]
    resume_lower = resume_text.lower()
    found_sections = sum(1 for ind in content_indicators if ind in resume_lower)
    content_score = round(min((found_sections / len(content_indicators)) * 100, 100), 1)

    return {
        "skills_score": skills_score,
        "keyword_score": keyword_score,
        "content_score": content_score
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "resume" not in request.files:
            return render_template("index.html", error="No resume file uploaded.")

        resume_file = request.files["resume"]
        jd_text = request.form.get("job_description", "").strip()

        if resume_file.filename == "":
            return render_template("index.html", error="Please select a resume file.")

        if not resume_file.filename.lower().endswith(".pdf"):
            return render_template("index.html", error="Only PDF files are supported.")

        if not jd_text:
            return render_template("index.html", error="Please enter a job description.")

        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            return render_template("index.html", error="Could not extract text from PDF.")

        # 🔍 Run analysis
        ats_score = compute_ats_score(resume_text, jd_text)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        missing_skills = get_missing_skills(resume_skills, jd_skills)
        resume_keywords = extract_top_keywords(resume_text, 15)
        jd_keywords = extract_top_keywords(jd_text, 15)
        missing_keywords = [kw for kw in jd_keywords if kw.lower() not in resume_text.lower()]

        resume_role, resume_conf = detect_role(resume_text)
        jd_role, jd_conf = detect_jd_role(jd_text)


        suggestions = generate_suggestions(
            missing_skills, missing_keywords, jd_role,
            resume_role, resume_skills, ats_score
        )

        resources = get_resource_links(missing_skills)

        breakdown = compute_breakdown(
            resume_text, jd_text,
            resume_skills, jd_skills,
            resume_keywords, jd_keywords
        )

        # ✅ SHOW RESUME BUILDER IF LOW SCORE
        show_resume_builder = ats_score < 50

        # 🎯 Score label
        if ats_score >= 75:
            score_label = "Excellent Match"
            score_color = "green"
        elif ats_score >= 50:
            score_label = "Good Match"
            score_color = "yellow"
        elif ats_score >= 25:
            score_label = "Fair Match"
            score_color = "orange"
        else:
            score_label = "Low Match"
            score_color = "red"

        # 📌 Role message
        if resume_role == jd_role:
            if ats_score >= 60:
                role_match_msg = f"✅ Strong match! Your resume fits the {jd_role} role well."
                role_match_type = "success"
            elif ats_score >= 30:
                role_match_msg = f"⚠️ Partial match for {jd_role}. Improve content & projects."
                role_match_type = "warning"
            else:
                role_match_msg = f"❌ Weak match for {jd_role}. Resume needs major improvement."
                role_match_type = "danger"
        elif jd_role != "General Professional":
            role_match_msg = (
                f"⚠️ Your resume is focused on '{resume_role}', but this job requires '{jd_role}'. "
                f"Try improving it using AI Resume Builder."
            )
            role_match_type = "warning"
        else:
            role_match_msg = (
                f"ℹ️ Your resume shows a '{resume_role}' profile. "
                f"Consider making it more role-specific."
            )
            role_match_type = "info"

        # 📦 Final data
        result_data = {
            "ats_score": ats_score,
            "score_label": score_label,
            "score_color": score_color,

            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "missing_skills": missing_skills,

            "resume_keywords": resume_keywords,
            "jd_keywords": jd_keywords,
            "missing_keywords": missing_keywords[:8],

            "resume_role": resume_role,
            "jd_role": jd_role,
            "role_match_msg": role_match_msg,
            "role_match_type": role_match_type,

            "suggestions": suggestions,
            "resources": resources,
            "breakdown": breakdown,

            "resume_conf": resume_conf,
            "jd_conf": jd_conf,

            # 🔥 NEW FEATURE DATA
            "show_resume_builder": show_resume_builder,
            "resume_text": resume_text,
            "jd_text": jd_text
        }

        session["result_data"] = result_data

        return render_template("result.html", data=result_data)

    except Exception as e:
        return render_template("index.html", error=f"Analysis failed: {str(e)}")

def rewrite_resume_ai(resume_text, jd_text):
    resume_text = resume_text.lower()
    jd_text = jd_text.lower()

    jd_skills = extract_skills(jd_text)

    # 🔹 Extract experience bullets (basic split)
    lines = resume_text.split("\n")
    experience_points = [line.strip() for line in lines if len(line.strip()) > 30]

    improved_points = []

    for point in experience_points:
        new_point = point

        # Add action words
        if "worked" in point:
            new_point = new_point.replace("worked", "Developed")

        if "made" in point:
            new_point = new_point.replace("made", "Built")

        # Add JD skills intelligently
        for skill in jd_skills[:3]:
            if skill.lower() not in new_point:
                new_point += f" using {skill}"

        improved_points.append(new_point.capitalize())

    # 🔹 Final Resume Format
    improved_resume = f"""
PROFESSIONAL SUMMARY
Results-driven professional aligned with job requirements. Skilled in {', '.join(jd_skills[:5])}.

SKILLS
{', '.join(jd_skills)}

EXPERIENCE
"""
    for p in improved_points[:6]:
        improved_resume += f"- {p}\n"

    improved_resume += f"""

PROJECTS
- Developed real-world projects aligned with job description
- Built dashboards, analysis reports, and automation tools

EDUCATION
(Your existing education)
"""

    return improved_resume


@app.route("/download")
def download():
    data = session.get("result_data")
    if not data:
        return "No analysis data found. Please analyze a resume first.", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Normal"],
                                  fontSize=24, textColor=colors.HexColor("#1e40af"),
                                  spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                     fontSize=11, textColor=colors.HexColor("#6b7280"),
                                     spaceAfter=20, alignment=TA_CENTER)
    section_style = ParagraphStyle("Section", parent=styles["Normal"],
                                    fontSize=14, textColor=colors.HexColor("#1e40af"),
                                    spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, textColor=colors.HexColor("#374151"),
                                 spaceAfter=4, leading=16)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"],
                                   fontSize=10, textColor=colors.HexColor("#374151"),
                                   spaceAfter=4, leftIndent=20, leading=16)

    # Title
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(Paragraph("Generated by AI Resume Analyzer", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e40af")))
    story.append(Spacer(1, 16))

    # ATS Score
    story.append(Paragraph("ATS Match Score", section_style))
    score_color_map = {"green": "#16a34a", "yellow": "#ca8a04", "orange": "#ea580c", "red": "#dc2626"}
    sc = score_color_map.get(data.get("score_color", "green"), "#16a34a")
    story.append(Paragraph(
        f'<font color="{sc}"><b>{data["ats_score"]}%</b></font> — {data["score_label"]}',
        ParagraphStyle("Score", parent=styles["Normal"], fontSize=18, spaceAfter=8)))

    # ATS Breakdown
    breakdown = data.get("breakdown", {})
    breakdown_table = [
        ["Metric", "Score"],
        ["Skills Score", f"{breakdown.get('skills_score', 0)}%"],
        ["Keyword Score", f"{breakdown.get('keyword_score', 0)}%"],
        ["Content Score", f"{breakdown.get('content_score', 0)}%"],
    ]
    t = Table(breakdown_table, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Role Detection
    story.append(Paragraph("Role Detection", section_style))
    story.append(Paragraph(f"<b>Resume Role:</b> {data.get('resume_role', 'N/A')}", body_style))
    story.append(Paragraph(f"<b>Job Role:</b> {data.get('jd_role', 'N/A')}", body_style))
    story.append(Paragraph(data.get("role_match_msg", ""), body_style))

    # Skills
    story.append(Paragraph("Skills Analysis", section_style))
    resume_skills_str = ", ".join(data.get("resume_skills", [])) or "None detected"
    missing_skills_str = ", ".join(data.get("missing_skills", [])) or "None — great coverage!"
    story.append(Paragraph(f"<b>Resume Skills:</b> {resume_skills_str}", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Missing Skills:</b> {missing_skills_str}", body_style))

    # Missing Keywords
    story.append(Paragraph("Missing Keywords", section_style))
    for kw in data.get("missing_keywords", []):
        story.append(Paragraph(f"• {kw}", bullet_style))
    if not data.get("missing_keywords"):
        story.append(Paragraph("No critical keywords missing.", body_style))

    # Suggestions
    story.append(Paragraph("AI Suggestions", section_style))
    for sug in data.get("suggestions", []):
        story.append(Paragraph(f"• {sug}", bullet_style))

    # Learning Resources
    story.append(Paragraph("Learning Resources", section_style))
    for res in data.get("resources", []):
        story.append(Paragraph(f"• <b>{res['skill']}</b>: {res['url']}", bullet_style))
    if not data.get("resources"):
        story.append(Paragraph("No specific resources needed — your skills coverage is strong!", body_style))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name="resume_analysis_report.pdf",
                     mimetype="application/pdf")



@app.route("/generate_resume", methods=["POST"])
def generate_resume():
    try:
        resume_text = request.form.get("resume_text", "")
        jd_text = request.form.get("jd_text", "")

        if not resume_text or not jd_text:
            return "Error: Missing data", 400

        # Extract analysis data
        jd_role, _ = detect_jd_role(jd_text)
        jd_skills = extract_skills(jd_text)
        resume_skills = extract_skills(resume_text)
        missing_skills = get_missing_skills(resume_skills, jd_skills)
        jd_keywords = extract_top_keywords(jd_text, 10)

        # Generate dynamic resume
        improved_resume = generate_dynamic_resume(
            resume_text, jd_text, jd_role, jd_skills, 
            resume_skills, missing_skills, jd_keywords
        )

        return render_template("generated_resume.html", 
                             resume=improved_resume,
                             role=jd_role,
                             skills=jd_skills)

    except Exception as e:
        return f"Error generating resume: {str(e)}", 500
    
if __name__ == "__main__":
    app.run(debug=False)
