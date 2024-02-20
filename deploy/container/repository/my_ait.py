#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
    requirements_generator.add_package('torch','2.1.0')
    requirements_generator.add_package('torchvision','0.16.0')
    requirements_generator.add_package('numpy','1.22.0')
    requirements_generator.add_package('scikit-learn', '1.3.1')
    requirements_generator.add_package('Pillow', '10.0.0')
    requirements_generator.add_package('pandas', '2.0.3')


# #### #3-3 [uneditable]

# In[5]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[6]:


import os
import pandas as pd
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold


# #### #4-2 [uneditable]

# In[7]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[8]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
    manifest_genenerator = AITManifestGenerator(current_dir)
    manifest_genenerator.set_ait_name('eval_correctness_image_classifier_pytorch')
    manifest_genenerator.set_ait_description('【機械学習モデルの正確性】を評価するため、データセットをランダムに分割し、それぞれの分割されたデータセットを対象としモデルで精度算出すること。その精度差が低ければ、モデルはデータセットに対し汎用的な性能を獲得していると判断すること。')
    manifest_genenerator.set_ait_source_repository('https://github.com/aistairc/Qunomon_AIT_eval_correctness_image_classifier_pytorch')
    manifest_genenerator.set_ait_version('1.0')
    manifest_genenerator.add_ait_licenses('Apache License Version 2.0')
    manifest_genenerator.add_ait_keywords('pytorch')
    manifest_genenerator.add_ait_keywords('Image Classification')
    manifest_genenerator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/C-1機械学習モデルの正確性')
    inventory_requirement_root_dir = manifest_genenerator.format_ait_inventory_requirement(format_=['*'])
    manifest_genenerator.add_ait_inventories(name='root_dir', 
                                             type_='dataset', 
                                             description='評価対象データセットのディレクトリ(すべて画像ファイルを一つフォルダに配置してください)', 
                                             requirement=inventory_requirement_root_dir)
    inventory_requirement_model = manifest_genenerator.format_ait_inventory_requirement(format_=['pt'])
    manifest_genenerator.add_ait_inventories(name='pytorch_model', 
                                             type_='model', 
                                             description='pytorchでトレーニング済みの画像分類モデル(モデルのアーキテクチャをつけて保存が必要 例:torch.save(model, モデル名称))', 
                                             requirement=inventory_requirement_model)
    inventory_requirement_label = manifest_genenerator.format_ait_inventory_requirement(format_=['csv'])
    manifest_genenerator.add_ait_inventories(name='label', 
                                             type_='attribute set', 
                                             description='評価対象データセットの画像ラベル値（CSVタイトル:image_path, lable）', 
                                             requirement=inventory_requirement_label)
    manifest_genenerator.add_ait_parameters(name='range_section', 
                                            type_='int', 
                                            description='データセット分割のサブセクション数', 
                                            default_val='5')
    manifest_genenerator.add_ait_parameters(name='calc_count', 
                                            type_='int', 
                                            description='試算回数', 
                                            default_val='5')
    manifest_genenerator.add_ait_measures(name='std_accuracy', 
                                          type_='float', 
                                          description='各回試算結果をまとめて算出された偏差値', 
                                          structure='single',
                                          min='0',
                                          max='1')
    manifest_genenerator.add_ait_downloads(name='calc_result', 
                                           description='各回試算結果の詳細')
    manifest_genenerator.add_ait_downloads(name='Log', 
                                           description='AIT実行ログ')
    manifest_path = manifest_genenerator.write()


# ### #6 Prepare for the Input

# [required]

# In[9]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_input_generator import AITInputGenerator
    input_generator = AITInputGenerator(manifest_path)
    input_generator.add_ait_inventories('root_dir','cifar_images')
    input_generator.add_ait_inventories('pytorch_model','resnet50.pt')
    input_generator.add_ait_inventories('label','cifar_labels.csv')
    input_generator.set_ait_params(name='range_section',
                                   value='5')
    input_generator.set_ait_params(name='calc_count',
                                   value='3')
    input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[10]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[11]:


@log(logger)
@measures(ait_output, 'std_accuracy')
def measure_acc(root_dir, model, label_path, label_to_int, range_section, calc_count):
    
    # Fomart dataset
    custom_dataset = CustomImageDataset(label_path=label_path, label_to_int=label_to_int, img_dir=root_dir)

    # Evaluate model stability
    std_deviation, eval_details_df = evaluate_model_stability(custom_dataset, model, range_section, calc_count)
    
    return std_deviation, eval_details_df

def evaluate_model_stability(dataset, model, n_splits, n_trials):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trial_accuracies = []
    calc_details = []

    for trial in range(n_trials):

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + trial)
        accuracies = []
        sub = 0
        for _, test_index in kf.split(dataset):
            print(f' -------------------- Trial {trial+1}-{sub+1} start -------------------- ')
            print(f'subset : {test_index}')
            print(f'KFold : {kf}')
            
            subset = Subset(dataset, test_index)
            print(f'subset : {subset}')
            print(f'subset length : {len(subset)}')
            loader = DataLoader(subset, batch_size=64, shuffle=True, collate_fn=lambda x: list(zip(*x)))
            
            accuracy = evaluate_model(model, loader, device)
            accuracies.append(accuracy)
            
            calc_details.append({'Trial': f'Trial {trial+1}-{sub+1}',
                                 'Subset': test_index,
                                 'accuracy': accuracy})

            print(f'\n　Accuracy: {accuracy}　\n')
            print(f' -------------------- Trial {trial+1}-{sub+1} end ---------------------- \n')
            sub += 1
        
        trial_accuracies.append(np.mean(accuracies))
    
    print(f'trial_accuracies: {trial_accuracies}')
    
    std_deviation = np.std(trial_accuracies)
    print(f'Accuracy standard deviation over trials: {std_deviation}')
    
    return std_deviation, pd.DataFrame(calc_details)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    idx = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = [transforms.functional.to_tensor(image).to(device) for image in images]
            images = torch.stack(images)
            labels = torch.tensor(labels).to(device)
            pred = model(images)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)
            print(f'batch run index:{idx}     correct:{correct}     total:{total} ')
            idx += 1
        accuracy = correct / total
    return accuracy


# In[12]:


class CustomImageDataset(Dataset):
    def __init__(self, label_path, label_to_int, img_dir):
        self.img_labels = pd.read_csv(label_path)
        self.img_dir = img_dir
        self.images = self.img_labels.iloc[:, 0].tolist()
        self.labels = [label_to_int[label] for label in self.img_labels.iloc[:, 1].tolist()]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path)
        label = self.labels[idx]
        return image, label


# In[13]:


@log(logger)
@downloads(ait_output, path_helper, 'calc_result', 'calc_result.csv')
def calc_result(results_df, file_path: str=None) -> str:    
    results_df.to_csv(file_path, index=False)


# In[14]:


@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None) -> str:
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[15]:


@log(logger)
@ait_main(ait_output, path_helper)
def main() -> None:

    # インベントリを読み込み
    root_dir = ait_input.get_inventory_path('root_dir')
    model = torch.load(ait_input.get_inventory_path('pytorch_model'))
    label_path = ait_input.get_inventory_path('label')
    df = pd.read_csv(label_path)
    unique_labels = df['label'].unique()
    label_to_int = {label: index for index, label in enumerate(unique_labels)}
    
    range_section = ait_input.get_method_param_value('range_section')
    calc_count = ait_input.get_method_param_value('calc_count')
    
    print('root_dir:', root_dir)
    print('model_path:', ait_input.get_inventory_path('pytorch_model'))
    print('label_path:', label_path)
    print('range_section:', range_section)
    print('calc_count:', calc_count)
    print('label_to_int:', label_to_int)
    
    print('\n        ************************************       \n')
                       
    std_deviation, calc_details_df = measure_acc(root_dir, model, label_path, label_to_int, range_section, calc_count)
    
    calc_result(calc_details_df)
    
    print('\n        ************************************       \n')

    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[ ]:


ait_owner='AIST'
ait_creation_year='2024'


# ### #12 Deployment

# [uneditable] 

# In[ ]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:




