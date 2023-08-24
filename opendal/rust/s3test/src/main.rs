use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use log::info;

use config::{Config, File};

use opendal::layers::LoggingLayer;
use opendal::services::S3;
use opendal::Operator;

#[tokio::main]
async fn main() -> Result<()> {
    println!("S3 test start");

    // Initialize our configuration reader
    let secrets = Config::builder()
        .add_source(File::from(Path::new("/Users/temp/Documents/playground/opendal/rust/s3test/secrets.toml")))
        .build()
        .unwrap();

    let access_key_id = secrets.get_string("GCS_ACCESS_KEY_ID").ok().unwrap();
    let secret_access_key = secrets.get_string("GCS_SECRET_ACCESS_KEY").ok().unwrap();
    // Print out our settings (as a HashMap)
    let debug_secrets = secrets.try_deserialize::<HashMap<String, String>>().unwrap();
    println!("secrets: {:?}", debug_secrets);

    let mut builder = S3::default();
    builder.root("o");
    builder.bucket("opendal-test");
    builder.region("us-west1");
    builder.endpoint("https://storage.googleapis.com/storage/v1/b");
    builder.access_key_id(&access_key_id);
    builder.secret_access_key(&secret_access_key);
    
    let op = Operator::new(builder)?
        .layer(LoggingLayer::default())
        .finish();
    info!("operator: {:?}", op);

    op.read("test.csv").await?;
    //op.write("hello.txt", "Hello, World!").await?;

    Ok(())
}
