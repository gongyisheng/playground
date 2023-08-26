mod gcs;
mod aws;

use std::io::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("S3 test start");
    aws::test_aws().await?;
    gcs::test_gcs().await?;
    println!("S3 test end");
    Ok(())
}
