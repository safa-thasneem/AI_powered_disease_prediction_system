# Generated by Django 3.2.25 on 2025-03-28 07:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('MYAPP', '0004_auto_20250214_1118'),
    ]

    operations = [
        migrations.CreateModel(
            name='history',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('diseases', models.CharField(max_length=100)),
                ('date', models.DateField(blank=True, null=True)),
                ('USER', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='MYAPP.registration')),
            ],
        ),
    ]
