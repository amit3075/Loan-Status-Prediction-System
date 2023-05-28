# from django.contrib.auth import authenticate, login
# from django.shortcuts import render, redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import numpy as np

from django.shortcuts import render ,HttpResponse,redirect
# import pickle
#
# with open('knnmodel.pkl', 'rb') as f:
#     mp = pickle.load(f)
# print(mp)
from joblib import load
from joblib._multiprocessing_helpers import mp

from Knn_test import knnaccu

mp = load('./model.joblib')


def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        print(username)
        print(pass1)
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('predict_output')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')



def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'SignupPage.html')



def predict_output(request):
    if request.method == 'POST':
        # Gender =int( request.POST['gender'])
        # Married = request.POST['married']
        # Graduate = request.POST['graduate']
        # Self_employed = request.POST['self_employed']
        Applicant_income = request.POST['applicant_income']
        Coapplicant_income = request.POST['coapplicant_income']
        Loan_amount = request.POST['loan_amount']

        Loan_amount_term = request.POST['loan_amount_term']
        print(Loan_amount_term)
        # Credit_history = request.POST['credit_history']
        # property_area = request.POST['property_area']
        user_input_list = [float(Applicant_income), int(Coapplicant_income), float(Loan_amount), int(Loan_amount_term)]
        print(user_input_list)
        if all(x == 0 for x in user_input_list):
            return render(request, 'index.html', {'output': 'Denied Loan'})
        # elif (Applicant_income * 0.5) == Loan_amount:
        #     return render (request,'index.html',{'output':'Denied Loan'})

        user_input_array = np.array([user_input_list])
        print(user_input_list)
        prediction = mp.predict(user_input_array)
        print(prediction)
        resultacc = str(knnaccu * 100)

        if prediction[0] == 0:
            return render(request, 'index.html', {'output': 'Denied Loan and accuracy' + resultacc})
        elif float(int(Applicant_income) / 2) < float(Loan_amount):
            return render(request, 'index.html', {'output': 'Denied Loan and accuracy' + resultacc})
        else:
            return render(request, 'index.html', {'output': 'Accept Loan and accuracy' + resultacc})

    return render(request, 'index.html')
