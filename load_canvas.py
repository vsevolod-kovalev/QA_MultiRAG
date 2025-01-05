from keys import CANVAS_DOMAIN, CANVAS_VSEVOLOD_KOVALEV
import requests

headers = {
    "Authorization": f"Bearer {CANVAS_VSEVOLOD_KOVALEV}"
}

def get_courses():
    url = f"{CANVAS_DOMAIN}/api/v1/courses"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()

def get_modules(course_id):
    url = f"{CANVAS_DOMAIN}/api/v1/courses/{course_id}/modules"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_files(course_id):
    url = f"{CANVAS_DOMAIN}/api/v1/courses/{course_id}/files"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

print(get_courses())
print(get_modules(1649995))

