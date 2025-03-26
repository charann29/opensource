from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import urllib.parse

app = Flask(__name__)

@app.route('/scrape-linkedin', methods=['GET'])
def scrape_linkedin():
    # Retrieve 'job_role' and 'location' from query parameters
    job_role = request.args.get('job_role', default='Frontend Developer', type=str)
    location = request.args.get('location', default='India', type=str)
    
    # Encode the parameters to ensure they are URL-safe
    job_role_encoded = urllib.parse.quote(job_role)
    location_encoded = urllib.parse.quote(location)
    
    # Construct the LinkedIn URL with the dynamic job role and location
    url = f'https://www.linkedin.com/jobs/search?keywords={job_role_encoded}&location={location_encoded}&pageNum=0'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        job_listings = soup.find_all('div', {'class': 'job-search-card'})
        
        jobs = []
        
        for job in job_listings:
            title = job.find('h3', {'class': 'base-search-card__title'}).text.strip()
            company = job.find('a', {'class': 'hidden-nested-link'}).text.strip()
            location = job.find('span', {'class': 'job-search-card__location'}).text.strip()
            anchor_tag = job.find('a', class_='base-card__full-link')
            href_link = anchor_tag['href']
            
            jobs.append({
                'title': title,
                'company': company,
                'location': location,
                'link': href_link
            })
        
        return jsonify(jobs)
    else:
        return jsonify({'error': 'Failed to fetch job listings'}), 500

if __name__ == '__main__':
    app.run(debug=True)
