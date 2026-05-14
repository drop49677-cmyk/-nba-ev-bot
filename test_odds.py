import os, requests
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('ODDS_API_KEY')
r = requests.get('https://api.the-odds-api.com/v4/sports/basketball_nba/events', params={'apiKey': api_key})
print('Events status:', r.status_code)
events = r.json()
print('Total events:', len(events))
if events:
    e_id = events[0]['id']
    print('Event:', events[0]['home_team'], 'vs', events[0]['away_team'])
    r2 = requests.get(f'https://api.the-odds-api.com/v4/sports/basketball_nba/events/{e_id}/odds', params={'apiKey': api_key, 'regions': 'us,eu', 'markets': 'player_points'})
    print('Odds status:', r2.status_code)
    print(str(r2.json())[:1000])
