mod gcs;
mod aws;

use std::io::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("S3 test start");
    gcs::test_gcs().await?;
    // aws::test_aws().await?;
    println!("S3 test end");
    Ok(())
}
