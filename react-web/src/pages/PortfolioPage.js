import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';

import Typography from '@material-ui/core/Typography';
import MessageDialog from '../components/MessageDialog'
import InvestmentStrategyRadios from '../components/InvestmentStrategyRadios'
import SubmitSelectionButton from '../components/SubmitSelectionButton';
import StockSelectSection from '../views/StockSelectSection'

import { BASEURL } from '../Constants';

const useStyles = makeStyles((theme) => ({
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
}));

function PortfolioPage() {
  const classes = useStyles();

  const [dataLoaded, setDataLoaded] = React.useState(false);
  const [companyData, setCompanyData] = React.useState([]);
  const [selectedStocks, setSelectedStocks] = React.useState([]);
  const [isMessageDialogOpen, setMessageDialogOpen] = React.useState(false);
  const [dialogTitle, setDialogTitle] = React.useState("Error");
  const [dialogMessage, setDialogMessage] = React.useState("");

  const handleMessageDialogOpen = () => {
    setMessageDialogOpen(true);
  };

  const handleMessageDialogClose = () => {
    setMessageDialogOpen(false);
  };

  const getCompanyData = async (e) => {
    const request = {
      method: 'GET',
    }

    try {
      console.log("Try to get company data");
      //setLoading(true);
      const response = await fetch(BASEURL + "/company", request)
      if (response.ok) {
        const jsonData = await response.json();
        if (jsonData.isSuccess) {
          setDataLoaded(true);
          setCompanyData(jsonData.data);
          console.log(jsonData.data[0]);
        } else {
          alert(jsonData.errorMsg);
          setDataLoaded(false);
        }
      }
    }
    catch (err) {
      setDataLoaded(false);
      console.log('Fetch company data failed', err);
    }
    finally {
      //setLoading(false);
    }
  };

  React.useEffect(() => {
    if (!dataLoaded) {
      getCompanyData();
    }
  }, [dataLoaded]);

  return (
    <div className={classes.root}>
      <MessageDialog
          isOpen={isMessageDialogOpen}
          handleClose={handleMessageDialogClose}
          title={dialogTitle}
          message={dialogMessage}
        >
      </MessageDialog>
      <Grid container direction="column" justify="center" alignItems="center">
        <Grid className={classes.title} item xs={12}>
          <Typography variant="h6">
            Choose your strategy
          </Typography>
        </Grid>
        <Grid className={classes.gridItem} item xs={12}>
          <InvestmentStrategyRadios />
        </Grid>
        <Grid className={classes.title} item xs={12}>
          <Typography variant="h6">
            Select Stocks
          </Typography>
        </Grid>
        <Grid className={classes.gridItem} item xs={12} >
          <StockSelectSection 
            selectedStocks={selectedStocks} 
            setSelectedStocks={setSelectedStocks} 
            companyData={companyData}
            setDialogMessage={setDialogMessage}
            openMessageDialog={handleMessageDialogOpen}
          >
          </StockSelectSection>
        </Grid>
        <Grid className={classes.gridItem} item xs={12} >
          <SubmitSelectionButton />
        </Grid>
      </Grid>
    </div>
  );
}

export default PortfolioPage;
