(ns xenon.prepare
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [mikera.image.core :as imagez]
            [xenon.parameters :as x-params]))

(defn- gather-files
  "Returns lazy seq of files in folder (including files in subfolders)"
  [folder]
  (filter #(.isFile %) (file-seq (io/file folder))))

(defn- split-files
  "Splits files into 2 collections"
  ([files]
   (split-files files 0.5))
  ([files split-perc]
   (partition-all (int (* split-perc (count files))) (shuffle files))))

(defn- file->label
  "Returns string with label based on file path"
  [file]
  (if (string/includes? (.getPath file) "positive")
    (first x-params/labels) (second x-params/labels)))

(defn- indexed-files-labels
  "Returns lazy seq of labaled files in format [indx file label]"
  [files]
  (map-indexed (fn [indx file] [indx file (file->label file)]) files))

(defn- save-img
  "Save resaized image in structure expected by RecordIO"
  [output-dir [indx file label]]
  (let [img-path (format "%s/%s/%05d.png" output-dir label indx)]
    (io/make-parents img-path)
    (-> (imagez/load-image file)
        (imagez/resize x-params/img-dim x-params/img-dim)
        (imagez/save img-path))))

(defn build-image-data
  "Saves processed images in structure expected by RecordIO:
  data
  ├── test
  │   ├── abnormal
  │   │   ├── 00000.png
  │   │   ├── 00001.png
  │   │   ├── ...
  │   │
  │   └── normal
  ├── train
  │   ├── abnormal
  │   └── normal
  └── val
      ├── abnormal
      └── normal"
  []
  (let [train-files            (gather-files x-params/train-data-dir)
        [test-files val-files] (split-files (gather-files x-params/valid-data-dir))]
    (println "Building" (count train-files) "training files")
    (dorun (pmap (partial save-img x-params/train-folder) (indexed-files-labels train-files)))
    (println "Building" (count test-files) "testing files")
    (dorun (pmap (partial save-img x-params/test-folder) (indexed-files-labels test-files)))
    (println "Building" (count val-files) "validation files")
    (dorun (pmap (partial save-img x-params/val-folder) (indexed-files-labels val-files)))))

(comment
  (build-image-data)

)
