import sys
sys.path.append(r'C:\\RVU\\Sem5\\student-complaint-intelligence')
from nlp_pipeline import classify_department

print('Result:', classify_department('the hostel food is very bad'))
