import os
import javalang
from tqdm import tqdm
from program_tasks.code_completion.util import create_tsv_file



def parse_java(src_folder, dest_dir, dest_file_name):
    with open(os.path.join(dest_dir, dest_file_name), 'w') as write_file:

        # judging if subfile is a directory
        if isinstance(src_folder, str):
            for f in os.listdir(src_folder):
                subfolder = os.path.join(src_folder, f)
                if os.path.isdir(subfolder): # train/project_name/java
                    print('tokenizing java code in {} ...'.format(subfolder))
                    for file_path in tqdm(os.listdir(subfolder)):
                        if file_path.endswith(".java"):
                            file = open(os.path.join(subfolder, file_path), 'r')
                            file_string = ' '.join(file.read().splitlines()) # read in oneline
                            tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                            token_str = " ".join([x.value for x in tokens])
                            # write new line each time, unicode escape
                            write_file.write(
                                token_str.encode('unicode_escape').decode('utf-8') + '\n')
                else: # project_name/java
                    if subfolder.endswith(".java"):
                        file = open(os.path.join(subfolder, file_path), 'r')
                        file_string = ' '.join(file.read().splitlines()) # read in oneline
                        tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                        token_str = " ".join([x.value for x in tokens])
                        # write new line each time, unicode escape
                        write_file.write(
                            token_str.encode('unicode_escape').decode('utf-8') + '\n')

            write_file.close()
             
        elif isinstance(src_folder, list):
            for src_f in src_folder:
                # print('tokenizing java code in {} ...'.format(src_f))
                for f in tqdm(os.listdir(src_f)):
                    subfolder = os.path.join(src_f, f)
                    if os.path.isdir(subfolder): # train/project_name/java
                        print('tokenizing java code in {} ...'.format(subfolder))
                        for file_path in tqdm(os.listdir(subfolder)):

                            if file_path.endswith(".java"):
                                file = open(os.path.join(src_f, file_path), 'r')
                                # read in oneline
                                file_string = ' '.join(file.read().splitlines()) 
                                # ast java parse
                                tokens = list(javalang.tokenizer.tokenize(file_string)) 
                                token_str = " ".join([x.value for x in tokens])
                                # write new line each time, unicode escape
                                write_file.write(
                                    token_str.encode('unicode_escape').decode('utf-8') + '\n'
                                ) 
                    else: # project_name/java
                        if subfolder.endswith(".java"):
                            file = open(os.path.join(subfolder, file_path), 'r')
                            file_string = ' '.join(file.read().splitlines()) # read in oneline
                            tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                            token_str = " ".join([x.value for x in tokens])
                            # write new line each time, unicode escape
                            write_file.write(
                                token_str.encode('unicode_escape').decode('utf-8') + '\n')

            write_file.close() 

        else:
            raise TypeError()



if __name__ == '__main__':
    # from preprocess.train_split import JAVA_PROJECTS

    # data_dir = "java_data/different_project/data"
    # java_dict = {
    #     'java_project1': {
    #         'train': os.path.join(data_dir, 'hibernate-orm', 'train'),
    #         'val': os.path.join(data_dir, 'hibernate-orm', 'val'),
    #         'test': os.path.join(data_dir, 'hadoop', 'val'),
    #     },
    #     'java_project2': {
    #         'train': os.path.join(data_dir, 'presto', 'train'),
    #         'val': os.path.join(data_dir, 'presto', 'val'),
    #         'test': os.path.join(data_dir, 'elasticsearch', 'val'),
    #     },
    #     'java_project3': {
    #         'train': os.path.join(data_dir, 'spring-framework', 'train'),
    #         'val': os.path.join(data_dir, 'spring-framework', 'val'),
    #         'test': os.path.join(data_dir, 'gradle', 'val'),
    #     },
    # }

    # for project in java_dict:

    #     dest_dir = "program_tasks/code_completion/dataset/different_project/"+project
    #     src_fn_dict = {
    #         'test.txt': java_dict[project]['test'],
    #         'val.txt': java_dict[project]['val'],
    #         'train.txt': java_dict[project]['train'],
    #     }
        
    #     if not os.path.exists(dest_dir):
    #         os.makedirs(dest_dir)

    #     for name, src in src_fn_dict.items():
    #         parse_java(src, dest_dir, name)

    #     for name in src_fn_dict:
    #         origin_file = os.path.join(dest_dir, name)
    #         dest_file = origin_file.rstrip('.txt') + '.tsv'
    #         create_tsv_file(origin_file, dest_file)

    # project = 'elasticsearch'
    # data_dir = 'java_data/different_author/' + project + '/data'
    # data_type = ['train', 'val', 'test1', 'test2', 'test3']
    # java_dict = {
    #     k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
    #     for k in data_type
    # }

    # dest_dir = "program_tasks/code_completion/dataset/different_author/" + project

    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)

    # for name, src in java_dict.items():
    #     parse_java(src, dest_dir, name)

    # for name in java_dict:
    #     origin_file = os.path.join(dest_dir, name)
    #     dest_file = origin_file.rstrip('.txt') + '.tsv'
    #     create_tsv_file(origin_file, dest_file)

    data_dir = 'java_data/different_time/data'
    data_type = ['train', 'val', 'test1', 'test2', 'test3']
    java_dict = {
        k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
        for k in data_type
    }

    dest_dir = "program_tasks/code_completion/dataset/different_time/java_project"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for name, src in java_dict.items():
        parse_java(src, dest_dir, name)

    for name in java_dict:
        origin_file = os.path.join(dest_dir, name)
        dest_file = origin_file.rstrip('.txt') + '.tsv'
        create_tsv_file(origin_file, dest_file)
    