from django.db import models

# Create your models here.AQ

class login_table(models.Model):
    username=models.CharField(max_length=250)
    password=models.CharField(max_length=100)
    type=models.CharField(max_length=100)


class registration(models.Model):
    LOGIN=models.ForeignKey(login_table,on_delete=models.CASCADE)
    name=models.CharField(max_length=250)
    email=models.CharField(max_length=100)
    pno=models.BigIntegerField()
    age=models.IntegerField()
    confirm_password=models.CharField(max_length=100)

class diseases(models.Model):
    disease = models.CharField(max_length=100)
    treatment=models.CharField(max_length=100)
    preventive_measure=models.CharField(max_length=100)


class symptoms(models.Model):
    DISEASE=models.ForeignKey(diseases,on_delete=models.CASCADE)
    symptoms=models.CharField(max_length=100)



class result(models.Model):
    DISEASE=models.ForeignKey(diseases,on_delete=models.CASCADE)
    result=models.CharField(max_length=100)

class history(models.Model):
    USER=models.ForeignKey(registration,on_delete=models.CASCADE)
    diseases=models.CharField(max_length=100)
    date=models.DateField(null=True,blank=True)





