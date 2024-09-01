from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from supabase import create_client, Client

app = Flask(__name__)

# Initialize Supabase client
url = "https://xpmrzwyanbomzkudoftu.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhwbXJ6d3lhbmJvbXprdWRvZnR1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQ2ODIzMjcsImV4cCI6MjA0MDI1ODMyN30.FCQm5HTwR7aglyGhw10lMEhE35t7pBNGTCyJvUWC2mk"
supabase: Client = create_client(url, key)

# Fetch data function
def fetch_data():
    freelancers_response = supabase.table('freelancers').select('*').execute()
    freelancers = pd.DataFrame(freelancers_response.data)
    jobs_response = supabase.table('jobs').select('*').execute()
    jobs = pd.DataFrame(jobs_response.data)
    return freelancers, jobs

# AI recommendation function
def recommend_jobs_with_ai(freelancer_id, freelancers, jobs):
    tfidf = TfidfVectorizer(stop_words='english')
    freelancer_tfidf_matrix = tfidf.fit_transform(freelancers['skills'])
    job_tfidf_matrix = tfidf.transform(jobs['job_description'])
    cosine_similarities = linear_kernel(freelancer_tfidf_matrix, job_tfidf_matrix)
    
    freelancer_idx = freelancers.index[freelancers['freelancer_id'] == freelancer_id].tolist()[0]
    sim_scores = list(enumerate(cosine_similarities[freelancer_idx]))
    
    def calculate_ai_score(similarity, job, freelancer):
        skill_match_score = len(set(freelancer.split(", ")) & set(job.split(", "))) / len(set(job.split(", ")))
        ai_score = 0.7 * similarity + 0.3 * skill_match_score
        return ai_score
    
    ai_scores = []
    for i, sim in sim_scores:
        job_desc = jobs.iloc[i]['job_description']
        freelancer_skills = freelancers.iloc[freelancer_idx]['skills']
        ai_score = calculate_ai_score(sim[1], job_desc, freelancer_skills)
        ai_scores.append((i, ai_score))
    
    ai_scores = sorted(ai_scores, key=lambda x: x[1], reverse=True)
    job_indices = [i[0] for i in ai_scores]
    return jobs.iloc[job_indices].to_dict(orient='records')

# Flask route
@app.route('/recommend', methods=['GET'])
def recommend():
    freelancer_id = request.args.get('freelancer_id', type=int)
    freelancers, jobs = fetch_data()
    recommendations = recommend_jobs_with_ai(freelancer_id, freelancers, jobs)
    return jsonify(recommendations)

# For Vercel to recognize the app
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run()
