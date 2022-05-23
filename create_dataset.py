from xlwt import Workbook
import string
import random




def create_xcel(emotion,video_title):
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1',cell_overwrite_ok=True)
    for i in range(0,len(emotion)):
        emo = emotion[i]
        for e in emo :
            sheet1.write(i,list(e.keys())[0] , list(e.values())[0])

    letters = string.ascii_lowercase
    name = ''.join(random.choice(letters) for i in range(10))
    name += ".xls"
    
    wb.save(name)