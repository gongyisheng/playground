for i in {1..100}
do 
    echo "Downloading $i"
    curl 172.31.92.214:8000/0000000000000000.data --output ~/0000000000000000.data
done
