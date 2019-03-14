(defproject xenon "0.1.0-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.clojure/tools.cli "0.3.7"]
                 [org.apache.clojure-mxnet/clojure-mxnet-linux-gpu "0.1.1-SNAPSHOT"]
                 [incanter "1.9.3"]
                 [net.mikera/imagez "0.12.0"]
                 [thinktopic/experiment "0.9.22"]
                 [quil "2.7.1"]]
  :plugins [[cider/cider-nrepl "0.17.0"]]
  :main xenon.core)
;  :jvm-opts ["-Djava.awt.headless=true"])
;             "-Dmxnet.traceLeakedObjects=true"
;             "-Dcom.sun.management.jmxremote"
;             "-Dcom.sun.management.jmxremote.authenticate=false"
;             "-Dcom.sun.management.jmxremote.ssl=false"
;             "-Dcom.sun.management.jmxremote.port=3134"])
