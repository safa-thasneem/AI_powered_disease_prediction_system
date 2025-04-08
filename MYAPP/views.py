from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# Create your views here.
from MYAPP.models import *
from MYAPP.predictMain import predict
import datetime
from django.contrib.auth.decorators import login_required

def login(request):
    return  render(request, 'index.html')

def login_post(request):
    print(request.POST)
    uname=request.POST['textfield']
    pswd=request.POST['textfield2']

    a=login_table.objects.filter(username=uname,password=pswd)
    if a.exists():
        b= login_table.objects.get(username=uname, password=pswd)
        if b.type == 'user':
           request.session["lid"]=b.id
           return HttpResponse('''<script>alert('user logined..');window.location='/user_home'</script>''')
        else:
           return HttpResponse('''<script>alert('falied..');window.location='/'</script>''')
    else:
        return HttpResponse('''<script>alert('falied..');window.location='/'</script>''')


def registration_page(request):
    return render(request,'registration/index.html')


def registration_post(request):
    name = request.POST['textfield']
    email = request.POST['textfield3']
    pno = request.POST['textfield4']
    age = request.POST['textfield7']
    username = request.POST['textfield2']
    password = request.POST['textfield6']
    confirm_password = request.POST['textfield5']

    # Check if password and confirm_password match
    if password == confirm_password:
        # Save login details to login_table
        login_instance = login_table()
        login_instance.username = username
        login_instance.password = password
        login_instance.type = 'user'
        login_instance.save()

        # Save registration details to registration table
        registration_instance = registration()
        registration_instance.name = name
        registration_instance.email = email
        registration_instance.pno = pno
        registration_instance.age = age
        registration_instance.password = password
        registration_instance.confirm_password = confirm_password
        registration_instance.LOGIN = login_instance
        registration_instance.save()

        # Registration success message
        return HttpResponse('''<script>alert('Registration successful!');window.location='/'</script>''')
    else:
        # Password mismatch error message
        return HttpResponse('''<script>alert('Error: Password and Confirm Password do not match.');window.location='/registration_page'</script>''')


from django.db.models import Count
def user_home(request):
    return render(request, 'HOME/index.html')

def history_home(request):
    distinct_diseases = history.objects.filter(USER__LOGIN=request.session["lid"]).values('diseases').annotate(count=Count('id')).order_by('-count')
    dict=[]
    # Printing the results
    for disease in distinct_diseases:
        print(f"Disease: {disease['diseases']}, Count: {disease['count']}")
        dict.append({"dis":disease['diseases'],"count":int(disease['count'])})

    return render(request, 'HOME/chartt.html',{"val":dict})


def image_upload(request):
    return render(request, 'HOME/file_upload.html', {"res": "upload image"})
def image_post(request):
    file=request.FILES["file"]
    fs=FileSystemStorage()
    fsave=fs.save(file.name,file)

    res=predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\media\/"+fsave)
    print(res[0],"kkkkkkkkkkkkkkkkkk")
    if res[0] == 'Breast Cancer':
        if res[1]=="Breast Benign":
            result="Take care....Monitor regularly, avoid unnecessary stress, and maintain a healthy lifestyle."
        else:
            result="Consult an oncologist, maintain a healthy diet, and follow up with regular mammograms.Seek immediate medical treatment, undergo further diagnosis, and follow prescribed therapy."
    elif res[0] =='Colon Cancer':
        if res[1]=="Colon Benign Tissue":
            result="Take care....Maintain a healthy diet and go for regular screenings."
        else:
            result="Undergo regular colonoscopies, maintain a fiber-rich diet, and consult a gastroenterologist.Seek immediate medical intervention and explore treatment options."
    elif res[0] =='Kidney Cancer':
        if res[1]=="Kidney Normal Tissue":
            result="Take care....No immediate action needed, continue with a healthy lifestyle."
        else:
            result="Increase water intake, consult a nephrologist, and consider treatment options.Get a biopsy for confirmation, consult a specialist, and explore treatment plans."
    elif res[0] =='Lung Cancer':
        if res[1]=="Lung Benign Tissue":
            result="Take care....No major concern but regular checkups are advised.No major concern but regular checkups are advised."
        elif res[1]=='Lung Adenocarcinoma':
            result="Consult an oncologist, follow targeted therapy, and maintain lung health."
        else:
            result="Avoid smoking, get medical imaging tests, and seek specialist advice.Follow chemotherapy/radiation treatments as advised by a doctor."
    else:
        result="No Recommendation"

    history_instance = history()
    history_instance.USER = registration.objects.get(LOGIN=request.session["lid"])
    history_instance.diseases = res[1]
    history_instance.date = datetime.datetime.now().date()
    history_instance.save()
    return render(request, 'HOME/file_upload.html', {"Mainclass":res[0], "Sub":res[1],"result":result})

def custom_logout_view(request):
    return redirect('/')