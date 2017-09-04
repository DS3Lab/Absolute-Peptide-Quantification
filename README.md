# OpenSwath and PyProphet Workflow Summary

1. Begin by running OpenSWATH on the dataset of mzXML or mzML files. To run OpenSWATH, build OpenMS and run OpenSwathWorkflow -

#####Convert all mzXML to mzML -

```
for file in /mnt/ds3lab/tiannan/ppp1/mzXML_files/*
do
FileConverter -in $file -out /mnt/ds3lab/tiannan/ppp1/mzML_converted_3/$(basename "$file").mzML
done
```

#####Run OpenSwathWorkflow on all mzXML files - 

```
for file in /mnt/ds3lab/tiannan/ppp1/mzXML_files/*
do
  OpenSwathWorkflow -min_upper_edge_dist 1 -mz_extraction_window 0.05 -rt_extraction_window 600 -extra_rt_extraction_window 100 -min_rsq 0.95 -min_coverage 0.6 -use_ms1_traces -Scoring:stop_report_after_feature 5 -tr_irt 15_Extracted_iRT_Decoy.TraML -tr 15_Extracted_Decoy.TraML -threads 16 -readOptions cache -tempDirectory /tmp -in $file -out_tsv OUT_TSV_$(basename "$file").tsv -out_chrom OUT_CHROM_$(basename "$file").chrom.mzML
done
```

#####Run OpenSwathWorkflow on all mzML files - 


```
for file in /mnt/ds3lab/tiannan/ppp1/mzML_converted_3/*
do
  OpenSwathWorkflow -min_upper_edge_dist 1 -mz_extraction_window 0.05 -rt_extraction_window 600 -extra_rt_extraction_window 100 -min_rsq 0.95 -min_coverage 0.6 -use_ms1_traces -Scoring:stop_report_after_feature 5 -tr_irt 15_Extracted_iRT_Decoy.TraML -tr 15_Extracted_Decoy.TraML -threads 24 -readOptions cache -tempDirectory tmp -in $file -out_tsv OUT_TSV_$(basename "$file").tsv -out_chrom OUT_CHROM_$(basename "$file").chrom.mzML
done
```

Where -tr_irt refers to the iRT library and -tr refers to the Retention Time Library, both of which need decoys appended to them, the libraries with decoys are under /libraries, for this experiment with the 15 AQUA peptides



2. Following the generation of the TSV and CHROM files, run pyProphet to get scores for the peak groups selected by OpenSwath -

#####Pyprophet-cli commands -

```
pyprophet-cli prepare --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder=/tmp/pyprophet_work/ --separator="tab" --extra-group-column="ProteinName"

pyprophet-cli subsample --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 1 --job-count 3 --sample-factor=0.4 &
pyprophet-cli subsample --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 2 --job-count 3 --sample-factor=0.4 &
pyprophet-cli subsample --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 3 --job-count 3 --sample-factor=0.4 &

pyprophet-cli learn --work-folder="/tmp/pyprophet_work/" --separator="tab" --ignore-invalid-scores


pyprophet-cli apply_weights --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 1 --job-count 3 &
pyprophet-cli apply_weights --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 2 --job-count 3 &
pyprophet-cli apply_weights --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --separator="tab" --job-number 3 --job-count 3 &

pyprophet-cli score --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --result-folder="/tmp/pyprophet_result_global" --separator="tab" \
--job-number 1 --job-count 3 --lambda=0.4 --statistics-mode=global --overwrite-results &
pyprophet-cli score --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --result-folder="/tmp/pyprophet_result_global" --separator="tab" \
--job-number 2 --job-count 3 --lambda=0.4 --statistics-mode=global --overwrite-results &
pyprophet-cli score --data-folder="/tmp/openswath_results/" --data-filename-pattern="*.tsv" \
--work-folder="/tmp/pyprophet_work/" --result-folder="/tmp/pyprophet_result_global" --separator="tab" \
--job-number 3 --job-count 3 --lambda=0.4 --statistics-mode=global --overwrite-results &
```


3. With the scores at hand, the peptide peak groups can now be extracted. The lower the m_score, the higher the rank assigned to a particular peak group. 

The data_peptide_complete.zip file contains all the extracted data needed to reconstruct a peak group, where each tuple is of the form (File Name, Peptide Group Label, Peptide ID, Start Time, Stop time, [Retention Time over ALL peak groups for that Peptide Group], [Intensity over ALL peak groups for that Peptide Group])

To get the values of intensity/RT for that particular peak group, the peak_fragment_annotation column has to be matched against the chrom.mzML output of OpenSwath and the binary data decoded for that fragment and plotted over the entire window. This is repeated for the peptide group and all fragments. Once the entire set of peak groups have been plotted, each peak group is selected individually by cutting the graph at the rightWidth and leftWidth portion. This is repeated for all peak groups across all swath files.

The peak groups can be extracted with the code snippets in the notebook `Peptide_Quantification.ipynb`.

With the extracted peak groups, AlexNet can be trained by running `finetune.py` after creating two text files with paths to the peak group images and their classes.

Clustering, grayscale map extraction and table creation can be done using the notebook `GrayscaleMapExtraction.ipynb`

The extracted peak group images/grayscale maps can then be used for further analysis.

The notebooks in this repository contain a lot of helper functions and snippets that was used for analysis, with some sample outputs.

A diary of all questions and progress over the period of this project is available on a google doc. You may have to request access for this. The scope and many tiny details are outlined clearly in this document.