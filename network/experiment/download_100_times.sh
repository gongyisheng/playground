for i in {1..100}
do 
    echo "Downloading $i"
    curl http://172.31.82.1:800 0/0000000000000000.data --output ~/0000000000000000.data
done
