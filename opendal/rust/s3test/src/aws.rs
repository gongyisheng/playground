use std::io::Result;
use std::path::Path;
use std::collections::HashMap;
use log::info;

use config::{Config, File};

use opendal::layers::LoggingLayer;
use opendal::services::S3;
use opendal::Operator;

pub async fn test_aws() -> Result<()> {
    println!("Testing AWS");
    // Initialize our configuration reader
    let secrets = Config::builder()
        .add_source(File::from(Path::new("/Users/temp/Documents/playground/opendal/rust/s3test/secrets.toml")))
        .build()
        .unwrap();

    let access_key_id = secrets.get_string("AWS_ACCESS_KEY_ID").ok().unwrap();
    let secret_access_key = secrets.get_string("AWS_SECRET_ACCESS_KEY").ok().unwrap();
    // Print out our settings (as a HashMap)
    // let debug_secrets = secrets.try_deserialize::<HashMap<String, String>>().unwrap();
    // println!("secrets: {:?}", debug_secrets);

    let mut builder = S3::default();
    builder.root("");
    builder.bucket("yisheng-backup");
    builder.region("us-east-1");
    builder.endpoint("https://s3.amazonaws.com");
    builder.access_key_id(&access_key_id);
    builder.secret_access_key(&secret_access_key);
    
    let op = Operator::new(builder)?
        .layer(LoggingLayer::default())
        .finish();
    info!("operator: {:?}", op);

    // read
    // let buf = op.read("hello/log4j-active (3).txt").await?;
    // let s = match std::str::from_utf8(&buf) {
    //     Ok(v) => v,
    //     Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
    // };
    // println!("s: {}", s);

    // write
    op.write("hello.txt", "Hello, World!").await?;

    // delete
    op.delete("hello.txt").await?;
    op.delete("test1.txt").await?; // should not raise 404

    // write
    op.write("test/test1.txt", "Hello, World!").await?;
    op.write("test/test2.txt", "Hello, World!").await?;

    // batch delete
    // op.remove(vec!["test/test1.txt".to_string(), "test/test2.txt".to_string()]).await?;
    op.remove_all("test/").await?;

    Ok(())
}