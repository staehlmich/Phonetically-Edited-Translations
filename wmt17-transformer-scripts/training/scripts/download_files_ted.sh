#!/bin/bash
#Modified by mstaehl
# Downloads IWSLT15 training and test data for EN-DE

script_dir=`dirname $0`
main_dir=$script_dir/..

# variables (toolkits; source and target language)
. $main_dir/vars

#MS: get EN-DE training data IWSLT15

#TO DO: Need to do something to download from Javascript!
# if [ ! -f $main_dir/downloads/de-en.tgz ];
# then
	# wget https://wit3.fbk.eu/download.php?release=2015-01&type=texts&slang=de&tlang=en -O $main_dir/downloads/de-en.tgz
	# tar -xf $main_dir/downloads/de-en.tgz -C $main_dir/downloads
# fi

 tar -xf $main_dir/downloads/de-en.tgz -C $main_dir/downloads

#MS: Test and dev set are in the same .tgz file (for TED-2015).

#MS: You need to use the copy command, otherwise you execute the script!
#MS: Copy training data to data directory

 cp $main_dir/downloads/de-en/train.tags.de-en.de  $main_dir/data/corpus_meta.de
 cp $main_dir/downloads/de-en/train.tags.de-en.en  $main_dir/data/corpus_meta.en

 #MS: create new directory for dev set and test set.
 mkdir $main_dir/data/dev/
 mkdir $main_dir/data/test/

 #MS: copy dev data to new directory.
 for year in 2010;
 do
	 cp $main_dir/downloads/de-en/IWSLT15.TED.dev${year}.de-en.de.xml  $main_dir/data/dev/IWSLT15.TED.dev$year.de-en.de.xml
	 cp $main_dir/downloads/de-en/IWSLT15.TED.dev${year}.de-en.en.xml  $main_dir/data/dev/IWSLT15.TED.dev$year.de-en.en.xml
 done

 #MS: Dev data TEDX
 for year in 2012;
 do
	 cp $main_dir/downloads/de-en/IWSLT15.TEDX.dev${year}.de-en.de.xml  $main_dir/data/dev/IWSLT15.TEDX.dev$year.de-en.de.xml
	 cp $main_dir/downloads/de-en/IWSLT15.TEDX.dev${year}.de-en.en.xml  $main_dir/data/dev/IWSLT15.TEDX.dev$year.de-en.en.xml
 done

 for year in {2010,2011,2012,2013};
 do
	 cp $main_dir/downloads/de-en/IWSLT15.TED.tst${year}.de-en.de.xml  $main_dir/data/test/IWSLT15.TED.tst$year.de-en.de.de-en.de.xml
	 cp $main_dir/downloads/de-en/IWSLT15.TED.tst${year}.de-en.en.xml  $main_dir/data/test/IWSLT15.TED.tst$year.de-en.de.de-en.en.xml

 done

 #MS: Test data TEDX
 for year in 2013;
 do
	 cp $main_dir/downloads/de-en/IWSLT15.TEDX.tst${year}.de-en.de.xml  $main_dir/data/test/IWSLT15.TEDX.tst$year.de-en.de.de-en.de.xml
	 cp $main_dir/downloads/de-en/IWSLT15.TEDX.tst${year}.de-en.en.xml  $main_dir/data/test/IWSLT15.TEDX.tst$year.de-en.de.de-en.en.xml
 done

 cd ..