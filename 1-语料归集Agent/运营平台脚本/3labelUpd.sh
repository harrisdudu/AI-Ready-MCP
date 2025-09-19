cp 3labelUpd.sql /data/NV/x86_corpus/corpusMiddleware/gbase8s/data/3labelUpd.sql
# cp 3labelUpd.sql /root/corpus/corpusMiddleware/gbase8s/data/3labelUpd.sql
docker exec -it corpus_gbase8s su - gbasedbt -c "dbaccess corpus_manage /opt/gbase/data/3labelUpd.sql"