#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
### Tutorial Documentation
Documentation that goes along with the Airflow tutorial located
[here](https://airflow.apache.org/tutorial.html)
"""
# [START tutorial]
# [START import_module]
from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from sqlalchemy import create_engine
import pandas as pd


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import io


# [END import_module]

# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model',
    default_args=default_args,
    description='A simple DAG for classificator benchmark calculation',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(0),
    tags=['example'],
)


class Model:
    """Класс, использующийся для прогона вычислений через нейросеть/любой другой преобразователь данных """

    def __init__(self, snapshot):
        """Инициализация модели. Это может быть загрузка нейросети в память GPU или любые другие подготовительные данные
           snapshot: снэпшот. В данном случае это может быть абсолютный путь до snapshots/mnist_cnn.pth,
           который содержит конфигурацию модели. Если снэпшота нет, равен None
        """

        # Загрузка модели
        self.model = Net()
        state = torch.load(snapshot, map_location=lambda storage, loc: storage)
        state = {key.replace('module.', ''): value for key, value in state.items()}
        self.model.load_state_dict(state)

        self.gpu_device = None
        self.model.eval()

    def map(self, samples):
        """
        Метод map работает с батчами, которые готовятся внутри ugraph и objprop.
        samples - словарь, где ключи - названия последовательностей входящих данных из input_columns,
        значения - python lists с данными, составляющие батч. В данном случае это список c подготовленными изображениям
        (которые прошли через функции decode_input, encode_input)
        """
        x = samples['image']
        x = np.stack(x, axis=0)
        with torch.no_grad():
            samples_face = torch.FloatTensor(x)
            if self.gpu_device is not None:
                samples_face = samples_face.cuda(device=self.gpu_device)
            result = self.model(samples_face).cpu().data.numpy()

        """
        Словарь, возвращаемый методом, сначала проходит через transform_output, encode_output, а затем записывается в файл.
        Нужно убедиться, что ключи словарей на всех этапах совпадают между собой.
        """
        return {'vector': result}

    @staticmethod
    def decode_input(sample: dict) -> dict:
        """Если на вход идет изображение, оно подается в функцию decode_input в виде байт-строки, которую нужно декодировать
           sample = {"image_enc": [bytes]  - одно изображение."""

        def decode_image(img_bytes):
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            return img

        result = []
        for img_enc in sample['image_enc']:
            img = decode_image(img_enc)
            result.append(img)

        return {'image': result}

    @staticmethod
    def transform_input(sample: dict) -> dict:

        """Преобразование матрицы картинки. Происходит после decode_input"""

        def transform_image(img):
            """Нормализация матрицы"""
            img = img.copy()
            img = (img - 0.1307) / 0.3081
            img = img[np.newaxis, :]
            return img

        result = []
        for img in sample['image']:
            img = transform_image(img)
            result.append(img)
        return {'image': result}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_test_data():
    images_prefix = "/usr/local/storage/benchmarks/mnist/trainingSample"
    images_list_file = "/usr/local/storage/benchmarks/mnist/content.txt"
    with open(images_list_file, "r") as file:
        images_list = [os.path.join(images_prefix, item.strip()) for item in file.readlines()]

    return images_list


def get_labels(images_list):
    labels = [os.path.basename(os.path.dirname(sample_name)) for sample_name in images_list]
    return np.array(labels).astype(int)


def search_new_model():
    new_path = "/usr/local/storage/new"
    current_path = "/usr/local/storage/current"
    new_items = os.listdir(new_path)
    print(new_items, new_path)
    if new_items:
        print("Found New {}, move into current dir".format(new_items))
        item = new_items[0]
        os.rename(os.path.join(new_path, item), os.path.join(current_path, item))
        print("Moved.")
        # os.remove(os.path.join(new_path, item))
    else:
        print("There is no new models, skip")


def run_model():
    images_prefix = "/usr/local/storage/benchmarks/mnist/trainingSample"
    images_list_file = "/usr/local/storage/benchmarks/mnist/content.txt"
    with open(images_list_file, "r") as file:
        images_list = [os.path.join(images_prefix, item.strip()) for item in file.readlines()]

    current_path = "/usr/local/storage/current"
    snapshots = os.listdir(current_path)
    if snapshots:
        snapshot_path = os.path.join(current_path, snapshots[0])
        model = Model(snapshot_path)
        samples = {"image_enc": [open(image_path, "rb").read() for image_path in images_list]}
        samples = model.transform_input(model.decode_input(samples))
        results = model.map(samples)

        print(results)
        np.save("/usr/local/storage/features/features__{}.npy".format(snapshots[0]), results["vector"])
    else:
        print("There is no current model, skip")


def run_metric_calculation():
    images_prefix = "/usr/local/storage/benchmarks/mnist/trainingSample"
    images_list_file = "/usr/local/storage/benchmarks/mnist/content.txt"
    with open(images_list_file, "r") as file:
        images_list = [os.path.join(images_prefix, item.strip()) for item in file.readlines()]
    labels = get_labels(images_list)
    features_dir = "/usr/local/storage/features"
    if os.listdir(features_dir):
        features_path = [os.path.join(features_dir, item) for item in os.listdir(features_dir)][0] 
        predictions = np.load(features_path)
        most_confident = predictions.argmax(axis=1)
        accuracy = np.mean(labels == most_confident)
        name = os.path.splitext(features_path)[0].split("__")[-1]
        engine = create_engine("postgresql+psycopg2://superset:superset@postgres_superset:5432/superset")
        df = pd.DataFrame({"name": [name], "result": [accuracy]})
        df.to_sql("benchmarks_results", engine, if_exists="append", index=False)
        os.remove(features_path)
        return accuracy

    else:
        print("There is no features path, no metric calculation")


def move_from_current():
    current_path = "/usr/local/storage/current"
    processed_path = "/usr/local/storage/processed"
    items = os.listdir(current_path)
    if items:
        print("There is model {} in current directory, moving".format(items))
        item = items[0]
        os.rename(os.path.join(current_path, item), os.path.join(processed_path, item))
        print("Moved.")
    else:
        print("There is no model in current directory")



t1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag,
)

t2 = PythonOperator(
    task_id="search_new",
    python_callable=search_new_model,
    dag=dag
)

t3 = PythonOperator(
    task_id='run_model',
    python_callable=run_model,
    dag=dag
)

t4 = PythonOperator(
    task_id='run_metric',
    python_callable=run_metric_calculation,
    dag=dag
)

t5 = PythonOperator(
    task_id='move_to_processed',
    python_callable=move_from_current,
    dag=dag
)

t1 >> t2 >> t3 >> t4 >> t5





