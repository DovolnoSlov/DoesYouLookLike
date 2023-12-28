FROM animcogn/face_recognition:latest

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip install --force-reinstall -v "aiogram==2.23.1"

COPY . .

CMD ["python", "app.py"]