use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use log::info;

use config::{Config, File};

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

    let ACCESS_KEY_ID = secrets.get_string(&"GCS_ACCESS_KEY_ID").ok().unwrap();
    let SECRET_ACCESS_KEY = secrets.get_string(&"GCS_SECRET_ACCESS_KEY").ok().unwrap();
    // Print out our settings (as a HashMap)
    let debug_secrets = secrets.try_deserialize::<HashMap<String, String>>().unwrap();
    println!("secrets: {:?}", debug_secrets);

    let mut builder = S3::default();
    builder.root("/Users/temp/Downloads");
    builder.bucket("opendal-test");
    builder.region("us-west1");
    builder.endpoint("https://storage.googleapis.com");
    builder.access_key_id(&ACCESS_KEY_ID);
    builder.secret_access_key(&SECRET_ACCESS_KEY);
    
    let op = Operator::new(builder)?.finish();
    info!("operator: {:?}", op);
    
    Ok(())
}
