cp 2labelUpd.sql /data/NV/x86_corpus/corpusMiddleware/gbase8s/data/2labelUpd.sql
docker exec -it corpus_gbase8s su - gbasedbt -c "dbaccess corpus_manage /opt/gbase/data/2labelUpd.sql"