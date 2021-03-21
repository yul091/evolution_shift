############### find all zip file that needs extrating

data_dir=/glusterfs/data/yxl190090/distribution_shift/data
target_folder=$data_dir/elasticsearch
cd $target_folder

zipfile=$(find -mindepth 0 -maxdepth 1 -name '*.zip') # find all zipfiles
echo "ALL zip files in the root dir: \n$zipfile"

for file in $zipfile
do 
    # first extract all files in the zip file
    file=${file#"./"} # get rid of the "./" prefix
    echo "extracting file $file ..." 
    unzip -q $file # -q to suppress the printing of extracted files

    # then copy the .java files to des folder
    file=${file%".zip"} # get rid of the ".zip" suffix
    echo "copying java files from $file ..." 
    cd $file

    des_dir=$target_folder/$file"_java"
    mkdir $des_dir
    java_list=$(find -name '*.java') # find all java files
    
    for javafile in $java_list
    do
        cp $javafile $des_dir # copy each java file to des folder
    done 

    cd .. # return to target_folder
done

####################################################################################################
######################################   elasticsearch   ###########################################
####################################################################################################

# cd $target_folder/elasticsearch-0.90.6
# echo "finding all java files in elasticsearch-0.90.6 ..."
# des_dir=$target_folder/elasticsearch_0.90.6
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/elasticsearch-2.3.2
# echo "finding all java files in elasticsearch-2.3.2 ..."
# des_dir=$target_folder/elasticsearch_2.3.2
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/elasticsearch-6.6.2
# echo "finding all java files in elasticsearch-6.6.2 ..."
# des_dir=$target_folder/elasticsearch_6.6.2
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/elasticsearch-7.11.1
# echo "finding all java files in elasticsearch-7.11.1 ..."
# des_dir=$target_folder/elasticsearch-7.11.1
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


####################################################################################################
##########################################   gradle   ##############################################
####################################################################################################

# cd $target_folder/gradle-5.3.0
# echo "finding all java files in gradle-5.3.0 ..."
# des_dir=$target_folder/gradle_5.3.0
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/gradle-6.8.3
# echo "finding all java files in gradle-6.8.3 ..."
# des_dir=$target_folder/gradle_6.8.3
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/gradle-REL_1.9-rc-4
# echo "finding all java files in gradle-REL_1.9-rc-4 ..."
# des_dir=$target_folder/gradle_REL_1.9-rc-4
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/gradle-REL_2.13
# echo "finding all java files in gradle-REL_2.13 ..."
# des_dir=$target_folder/gradle_REL_2.13
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


####################################################################################################
#########################################   wildfly   ##############################################
####################################################################################################


# # cd /glusterfs/data/yxl190090/distribution_shift/dataset/wildfly-16.0.0.Beta1
# cd $target_folder/wildfly-10.1.0.CR1
# echo "finding all java files in wildfly-10.1.0.CR1 ..."
# des_dir='wildfly_10.1.0.CR1'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-10.1.0.CR1/wildfly_10.1.0.CR1 .


# cd $target_folder/wildfly-17.0.0.Alpha1
# echo "finding all java files in wildfly-17.0.0.Alpha1 ..."
# des_dir='wildfly_17.0.0.Alpha1'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-17.0.0.Alpha1/wildfly_17.0.0.Alpha1 .


# cd $target_folder/wildfly-23.0.0.Beta1
# echo "finding all java files in wildfly-23.0.0.Beta1 ..."
# des_dir='wildfly_23.0.0.Beta1'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-23.0.0.Beta1/wildfly_23.0.0.Beta1 .


# cd $target_folder/wildfly-8.0.0.Beta1
# echo "finding all java files in wildfly-8.0.0.Beta1 ..."
# des_dir='wildfly_8.0.0.Beta1'
# java_list=$(find -name '*.java')
# mkdir $des_dir

# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done
# cd ..
# mv wildfly-8.0.0.Beta1/wildfly_8.0.0.Beta1 .


####################################################################################################
##########################################   hadoop   ##############################################
####################################################################################################

# cd $target_folder/hadoop-ozone-0.4.0-alpha-RC2
# echo "finding all java files in hadoop-ozone-0.4.0-alpha-RC2 ..."
# des_dir=$target_folder/hadoop-ozone_0.4.0-alpha-RC2
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/hadoop-release-2.2.0
# echo "finding all java files in hadoop-release-2.2.0 ..."
# des_dir=$target_folder/hadoop-release_2.2.0
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/hadoop-rel-release-3.2.2
# echo "finding all java files in hadoop-rel-release-3.2.2 ..."
# des_dir=$target_folder/hadoop-rel-release_3.2.2
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/hadoop-YARN-2928-2016-06-21
# echo "finding all java files in hadoop-YARN-2928-2016-06-21 ..."
# des_dir=$target_folder/hadoop-YARN_2928-2016-06-21
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


####################################################################################################
######################################   hibernate-orm   ###########################################
####################################################################################################

# cd $target_folder/hibernate-orm-4.2.23.Final
# echo "finding all java files in hibernate-orm-4.2.23.Final ..."
# des_dir=$target_folder/hibernate-orm_4.2.23.Final
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # copy all java files in one folder
# for javafile in $java_list
# do
#     cp $javafile $des_dir
# done


# cd $target_folder/hibernate-orm-4.3.0.CR1
# echo "finding all java files in hibernate-orm-4.3.0.CR1 ..."
# des_dir=$target_folder/hibernate-orm_4.3.0.CR1
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # copy all java files in one folder
# for javafile in $java_list
# do
#     cp $javafile $des_dir
# done


# cd $target_folder/hibernate-orm-5.3.10
# echo "finding all java files in hibernate-orm-5.3.10 ..."
# des_dir=$target_folder/hibernate-orm_5.3.10
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # cp all java files in one folder
# for javafile in $java_list
# do
#     cp $javafile $des_dir
# done


# cd $target_folder/hibernate-orm-5.4.29
# echo "finding all java files in hibernate-orm-5.4.29 ..."
# des_dir=$target_folder/hibernate-orm_5.4.29
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # cp all java files in one folder
# for javafile in $java_list
# do
#     cp $javafile $des_dir
# done


####################################################################################################
#########################################   presto   ###############################################
####################################################################################################

# cd $target_folder/presto-0.53
# echo "finding all java files in presto-0.53 ..."
# des_dir=$target_folder/presto_0.53
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/presto-0.145
# echo "finding all java files in presto-0.145 ..."
# des_dir=$target_folder/presto_0.145
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/presto-0.220
# echo "finding all java files in presto-0.220 ..."
# des_dir=$target_folder/presto_0.220
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/presto-0.248
# echo "finding all java files in presto-0.248 ..."
# des_dir=$target_folder/presto_0.248
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


####################################################################################################
####################################   spring-framework   ##########################################
####################################################################################################

# cd $target_folder/spring-framework-3.2.5.RELEASE
# echo "finding all java files in spring-framework-3.2.5.RELEASE ..."
# des_dir=$target_folder/spring-framework_3.2.5.RELEASE
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/spring-framework-3.2.17.RELEASE
# echo "finding all java files in spring-framework-3.2.17.RELEASE ..."
# des_dir=$target_folder/spring-framework_3.2.17.RELEASE
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/spring-framework-5.2.0.M2
# echo "finding all java files in spring-framework-5.2.0.M2 ..."
# des_dir=$target_folder/spring-framework_5.2.0.M2
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done


# cd $target_folder/spring-framework-5.3.4
# echo "finding all java files in spring-framework-5.3.4 ..."
# des_dir=$target_folder/spring-framework_5.3.4
# java_list=$(find -name '*.java')
# mkdir $des_dir
# # move all java files in one folder
# for javafile in $java_list
# do
#     mv $javafile $des_dir
# done



####################################################################################################
#################################   create train/test/valid   ######################################
####################################################################################################

# cd $target_folder
# mkdir train
# mkdir test1
# mkdir test2
# mkdir test3
# # mkdir test4

# mv gradle-REL_1.9-rc-4 train
# # mv gradle_5.2.0 train
# # mv wildfly_16.0.0.Beta1 train

# mv gradle-REL_2.13 test1
# mv gradle-5.3.0 test2
# mv gradle-6.8.3 test3
# # mv hadoop_3.2.2 test4

