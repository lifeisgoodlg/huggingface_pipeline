import os
import re
import pandas
import warnings 
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

warnings.filterwarnings('ignore')

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

reg_text = re.compile('<.*?>')

def get_video_id(url):
    pattern = r'(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/|shorts/))([^#&?]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def clean_comment(comment):
  cleaned_comment = re.sub(reg_text, '', comment)
  return cleaned_comment


def save_comment(url):
    video_id = get_video_id(url)

    comments = list()
    api_obj = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, maxResults=100).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append(clean_comment(comment['textDisplay'])) 
    
            if item['snippet']['totalReplyCount'] > 0 and 'replies' in item:  
                for reply_item in item['replies']['comments']:
                    reply = reply_item['snippet']
                    comments.append(clean_comment(reply['textDisplay']))
    
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break

    return comments

# df = pandas.DataFrame(comments)
# df.to_csv('results.csv', header=['comment', 'author', 'date', 'num_likes'], index=None)