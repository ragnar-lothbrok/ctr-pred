import requests
from mt103 import MT103
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sklearn
# import csv

fileName = "testing.csv"
writer = open(fileName, 'w')
# writer = open("csvexample31.csv", "w")
# writer = csv.writer(myFile)
# myFields = ['status','bank_operation_code', 'beneficiary', 'day', 'month', 'year', 'details_of_charges',
#             'interbank_settled_amount', 'interbank_settled_currency', 'intermediary', 'ordering_customer',
#             'ordering_institution', 'receiver_correspondent', 'regulatory_reporting', 'remittance_information',
#             'sender_correspondent', 'sender_to_receiver_information', 'transaction_reference']
# myFields = ['status','bank_operation_code', 'beneficiary', 'day', 'month', 'year',
#             'interbank_settled_amount', 'interbank_settled_currency', 'intermediary', 'ordering_customer',
#             'ordering_institution', 'receiver_correspondent', 'regulatory_reporting', 'remittance_information',
#             'sender_correspondent', 'sender_to_receiver_information', 'transaction_reference']
# myFields = ['status','bank_operation_code', 'beneficiary', 'day', 'month', 'year', 'intermediary', 'ordering_customer',
#             'ordering_institution', 'receiver_correspondent', 'regulatory_reporting', 'remittance_information',
#             'sender_correspondent', 'sender_to_receiver_information', 'transaction_reference']
myFields = ['status','bank_operation_code', 'beneficiary', 'day', 'month', 'year',  'ordering_customer',
            'ordering_institution']
# writer = csv.DictWriter(myFile, fieldnames=myFields)
record = "";
for value in myFields:
    record = record + ","+str(value);
# writer.writeheader()
writer.write(record[1:len(record)])
writer.write("\n")
file = open("/Users/coder/Downloads/TestingData.csv", "r")
for line in file:
    print(line)
    data = line.split("|")
    if len(line) < 10:
        break
    if "score" in  data[1]:
        continue
    mt103 = MT103(data[3])
    print("=============")
    print(mt103.text)
    print("======****=======")
    # my_dict = {}
    # my_dict['bank_operation_code'] = mt103.text.bank_operation_code
    # my_dict['beneficiary'] = mt103.text.beneficiary
    # my_dict['day'] = mt103.text.date.day
    # my_dict['month'] = mt103.text.date.month
    # my_dict['year'] = mt103.text.date.year
    # my_dict['details_of_charges'] = mt103.text.details_of_charges
    # my_dict['interbank_settled_amount'] = mt103.text.interbank_settled_amount
    # my_dict['interbank_settled_currency'] = mt103.text.interbank_settled_currency
    # my_dict['intermediary'] = mt103.text.intermediary
    # my_dict['ordering_customer'] = mt103.text.ordering_customer
    # my_dict['ordering_institution'] = mt103.text.ordering_institution
    # my_dict['receiver_correspondent'] = mt103.text.receiver_correspondent
    # my_dict['regulatory_reporting'] = mt103.text.regulatory_reporting
    # my_dict['remittance_information'] = mt103.text.remittance_information
    # my_dict['sender_correspondent'] = mt103.text.sender_correspondent
    # my_dict['sender_to_receiver_information'] = mt103.text.sender_to_receiver_information
    # my_dict['transaction_reference'] = mt103.text.transaction_reference
    #
    # print("======DONE=======")


    list = []
    if data[0] == 'Blocked':
        list.append(0)
    elif data[0] == 'False Hit':
        list.append(0.5)
    else:
        list.append(1)

    # list.append(data[0])
    if hasattr(mt103.text,'bank_operation_code'):
        if mt103.text.bank_operation_code is None:
            list.append("NA")
        else:
            list.append(mt103.text.bank_operation_code)
    else:
        list.append("NA")

    if hasattr(mt103.text,'beneficiary'):
        if mt103.text.beneficiary is None:
            list.append("NA")
        else:
            list.append(mt103.text.beneficiary)
    else:
        list.append("NA")


    if hasattr(mt103.text,'date'):
        if mt103.text.date is None:
            list.append("0")
            list.append("0")
            list.append("0")
        else:
            list.append(mt103.text.date.day)
            list.append(mt103.text.date.month)
            list.append(mt103.text.date.year)
    else:
        list.append("0")
        list.append("0")
        list.append("0")


    # list.append(mt103.text.details_of_charges)

    # if hasattr(mt103.text,'interbank_settled_amount'):
    #     if mt103.text.interbank_settled_amount is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.interbank_settled_amount)
    # else:
    #     list.append("NA")
    #
    # # list.append(mt103.text.interbank_settled_amount)
    #
    # if hasattr(mt103.text,'interbank_settled_currency'):
    #     if mt103.text.interbank_settled_currency is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.interbank_settled_currency)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.interbank_settled_currency)


    # if hasattr(mt103.text, 'intermediary'):
    #     if mt103.text.intermediary is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.intermediary)
    # else:
    #     list.append("NA")
    # list.append(mt103.text.intermediary)


    if hasattr(mt103.text,'ordering_customer'):
        if mt103.text.ordering_customer is None:
            list.append("NA")
        else:
            list.append(mt103.text.ordering_customer)
    else:
        list.append("NA")

    # list.append(mt103.text.ordering_customer)

    if hasattr(mt103.text,'ordering_institution'):
        if mt103.text.ordering_institution is None:
            list.append("NA")
        else:
            list.append(mt103.text.ordering_institution)
    else:
        list.append("NA")

    # list.append(mt103.text.ordering_institution)

    # if hasattr(mt103.text,'receiver_correspondent'):
    #     if mt103.text.receiver_correspondent is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.receiver_correspondent)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.receiver_correspondent)


    # if hasattr(mt103.text,'regulatory_reporting'):
    #     if mt103.text.regulatory_reporting is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.regulatory_reporting)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.regulatory_reporting)

    # if hasattr(mt103.text,'remittance_information'):
    #     if mt103.text.remittance_information is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.remittance_information)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.remittance_information)

    # if hasattr(mt103.text,'sender_correspondent'):
    #     if mt103.text.sender_correspondent is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.sender_correspondent)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.sender_correspondent)


    # if hasattr(mt103.text,'sender_to_receiver_information'):
    #     if mt103.text.sender_to_receiver_information is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.sender_to_receiver_information)
    # else:
    #     list.append("NA")

    # list.append(mt103.text.sender_to_receiver_information)

    # if hasattr(mt103.text,'transaction_reference'):
    #     if mt103.text.transaction_reference is None:
    #         list.append("NA")
    #     else:
    #         list.append(mt103.text.transaction_reference)
    # else:
    #     list.append("NA")
    # list.append(mt103.text.transaction_reference)
    record = "";
    for value in list:
        record = record + ","+str(value).replace(",", "").replace("X","").replace("/","");

    writer.write(record[1:len(record)])
    writer.write("\n")



amlpd=pd.read_csv(fileName,
                  header=0, skiprows=0)

# amlpd = amlpd.sample(frac=1)

amlpd = sklearn.utils.shuffle(amlpd)

amlpd= amlpd.head(1000)

# print(amlpd)

graph1 = amlpd[['year','status']]

amlpd.plot(style=".")
plt.show()

my_plot = graph1.plot(kind='bar')
plt.show()