############### find all zip file that needs extrating
cd /glusterfs/data/yxl190090/distribution_shift
# ls # list of files and folders in the project

zipfile=$(find -mindepth 0 -maxdepth 1 -name '*.zip') # find all zipfiles
echo "ALL zip files in the root dir: $zipfile"

for file in $zipfile
do 
    echo "extracting file: ${file#"./"} ..." 
    unzip $file -d 'dataset'
    # find . -name ${file#"./"} -type f -exec unzip -jd "dataset/${file%".zip"}/{}" "{}" "*.java" \;
done

####################################################################################################
######################################   elasticsearch   ###########################################
####################################################################################################


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/elasticsearch-6.6.0
# echo "finding all java files in elasticsearch-6.6.0 ..."
# des_dir='java_files_6.6.0'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv elasticsearch-6.6.0/java_files_6.6.0 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/elasticsearch-7.6.0
# echo "finding all java files in elasticsearch-7.6.0 ..."
# des_dir='java_files_7.6.0'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv elasticsearch-7.6.0/java_files_7.6.0 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/elasticsearch-7.11.0
# echo "finding all java files in elasticsearch-7.11.0 ..."
# des_dir='java_files_7.11.0'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv elasticsearch-7.11.0/java_files_7.11.0 .


####################################################################################################
##########################################   gradle   ##############################################
####################################################################################################


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/gradle-5.2.0
# echo "finding all java files in gradle-5.2.0 ..."
# des_dir='gradle_5.2.0'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv gradle-5.2.0/gradle_5.2.0 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/gradle-6.2.0
# echo "finding all java files in gradle-6.2.0 ..."
# des_dir='gradle_6.2.0'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv gradle-6.2.0/gradle_6.2.0 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/gradle-6.8.2
# echo "finding all java files in gradle-6.8.2 ..."
# des_dir='gradle_6.8.2'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv gradle-6.8.2/gradle_6.8.2 .



####################################################################################################
#########################################   wildfly   ##############################################
####################################################################################################


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/wildfly-16.0.0.Beta1
# echo "finding all java files in wildfly-16.0.0.Beta1 ..."
# des_dir='wildfly_16.0.0.Beta1'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-16.0.0.Beta1/wildfly_16.0.0.Beta1 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/wildfly-19.0.0.Beta2
# echo "finding all java files in wildfly-19.0.0.Beta2 ..."
# des_dir='wildfly_19.0.0.Beta2'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-19.0.0.Beta2/wildfly_19.0.0.Beta2 .


# cd /glusterfs/data/yxl190090/distribution_shift/dataset/wildfly-22.0.1.Final
# echo "finding all java files in wildfly-22.0.1.Final ..."
# des_dir='wildfly_22.0.1.Final'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-22.0.1.Final/wildfly_22.0.1.Final .


####################################################################################################
##########################################   hadoop   ##############################################
####################################################################################################


cd /glusterfs/data/yxl190090/distribution_shift/dataset/hadoop-rel-release-3.1.2
echo "finding all java files in hadoop-rel-release-3.1.2 ..."
des_dir='hadoop_3.1.2'
java_list=$(find -name '*.java')
mkdir $des_dir

# move all java files in one folder
for javafile in $java_list
do
    mv $javafile $des_dir
done
cd ..
mv hadoop-rel-release-3.1.2/hadoop_3.1.2 .


cd /glusterfs/data/yxl190090/distribution_shift/dataset/hadoop-release-3.1.3-RC0
echo "finding all java files in hadoop-release-3.1.3-RC0 ..."
des_dir='hadoop_3.1.3'
java_list=$(find -name '*.java')
mkdir $des_dir

# move all java files in one folder
for javafile in $java_list
do
    mv $javafile $des_dir
done
cd ..
mv hadoop-release-3.1.3-RC0/hadoop_3.1.3 .


cd /glusterfs/data/yxl190090/distribution_shift/dataset/hadoop-release-3.1.4-RC0
echo "finding all java files in hadoop-release-3.1.4-RC0 ..."
des_dir='hadoop_3.1.4'
java_list=$(find -name '*.java')
mkdir $des_dir

# move all java files in one folder
for javafile in $java_list
do
    mv $javafile $des_dir
done
cd ..
mv hadoop-release-3.1.4-RC0/hadoop_3.1.4 .



cd /glusterfs/data/yxl190090/distribution_shift/dataset/hadoop-rel-release-3.2.2
echo "finding all java files in hadoop-rel-release-3.2.2 ..."
des_dir='hadoop_3.2.2'
java_list=$(find -name '*.java')
mkdir $des_dir

# move all java files in one folder
for javafile in $java_list
do
    mv $javafile $des_dir
done
cd ..
mv hadoop-rel-release-3.2.2/hadoop_3.2.2 .

####################################################################################################
#################################   create train/test/valid   ######################################
####################################################################################################

cd /glusterfs/data/yxl190090/distribution_shift/dataset
mkdir train
mkdir test1
mkdir test2
mkdir test3
mkdir test4

mv java_files_6.6.0 train
mv gradle_5.2.0 train
mv wildfly_16.0.0.Beta1 train

mv hadoop_3.1.2 test1
mv hadoop_3.1.3 test2
mv hadoop_3.1.4 test3
mv hadoop_3.2.2 test4

