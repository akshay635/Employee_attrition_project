# config.py
MEDIANS_PATH = 'employee_attrition_project/data/medians.json'
MODES_PATH = 'employee_attrition_project/data/modes.json'

# common selected features 
COMMON_FEATURES = ['Environment_Satisfaction', 'Salary_Hike_in_percent', 
                   'Salary', 'Job_Involvement', 'Years_since_last_promotion',
                   'Age', 'Overtime', 'Job_Satisfaction', 'Business_Travel',
                   'Distance_From_Home', 'Work_life_balance', 'Department', 'Job_Role']

# Business Travel
BT_FEATURES = ['Travel Rarely', 'No Travel', 'Travel Frequently']

# IT FIELDS
SOFTWARE_FIELDS = ['Software Development', 'Cyber Security', 'Data Science', 
                   'Network Administration', 'IT Services']

# SOFTWARE ROLES
SOFTWARE_ROLES = ['Developer', 'Software Engineer', 'IT', 'Technician', 
                  'Support', 'Consultant', 'Director', 'HR', 'Help Desk', 
                  'QA Analyst', 'Manager', 'Business Analyst']

# TARGET variable 
TARGET = ['']

# OVERTIME ('YES' OR 'NO')
OVERTIME = ['Yes', 'No']


