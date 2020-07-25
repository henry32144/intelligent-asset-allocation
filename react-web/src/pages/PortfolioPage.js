import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import MessageDialog from '../components/MessageDialog'
import StockSelectSection from '../views/StockSelectSection'
import PortfolioToolBar from '../components/PortfolioToolBar'
import { BASEURL } from '../Constants';

const useStyles = makeStyles((theme) => ({
  portfolioPage: {
    height: 'calc(100% - 56px)',
    [`${theme.breakpoints.up('xs')} and (orientation: landscape)`]: {
      height: 'calc(100% - 48px)',
    },
    [theme.breakpoints.up('sm')]: {
      height: 'calc(100% - 64px)',
    },
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
  portfolioContent: {
    flex: 'auto'
  }
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
    <div className={classes.portfolioPage}>
      <MessageDialog
        isOpen={isMessageDialogOpen}
        handleClose={handleMessageDialogClose}
        title={dialogTitle}
        message={dialogMessage}
      >
      </MessageDialog>
      <PortfolioToolBar
        selectedStocks={selectedStocks}
        setSelectedStocks={setSelectedStocks}
        companyData={companyData}
        setDialogMessage={setDialogMessage}
        openMessageDialog={handleMessageDialogOpen}
      >
      </PortfolioToolBar>
      <Grid item container direction="row" justify="flex-start" alignItems="stretch" className={classes.portfolioContent}>
        <Grid item xs={9} sm={3} >
          <StockSelectSection
            selectedStocks={selectedStocks}
            setSelectedStocks={setSelectedStocks}
          >
          </StockSelectSection>
        </Grid>
      </Grid>
    </div>
  );
}

export default PortfolioPage;