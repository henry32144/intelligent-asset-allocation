import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import MessageDialog from '../components/MessageDialog'
import CreatePortfolioDialog from '../components/CreatePortfolioDialog'
import StockSelectSection from '../views/StockSelectSection'
import PortfolioToolBar from '../components/PortfolioToolBar'
import NewsSection from '../views/NewsSection'
import Typography from '@material-ui/core/Typography';
import { motion } from "framer-motion"
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
  },
  companyName: {
    margin: theme.spacing(4, 0, 2),
  },
  sideBar: {
    zIndex: 1400,
  }
}));

function PortfolioPage(props) {
  const classes = useStyles();

  const [companyData, setCompanyData] = React.useState([]);
  const [userPortfolios, setUserPortfolios] = React.useState([]); 
  const [selectedStocks, setSelectedStocks] = React.useState([]);
  const [dataLoaded, setDataLoaded] = React.useState(false);
  const [isSideBarExpanded, setSideBarExpand] = React.useState(false);
  const [isMessageDialogOpen, setMessageDialogOpen] = React.useState(false);
  const [isCreatePortfolioDialogOpen, setCreatePortfolioDialogOpen] = React.useState(false);
  const [dialogTitle, setDialogTitle] = React.useState("");
  const [dialogMessage, setDialogMessage] = React.useState("");
  const [currentSelectedPortfolio, setCurrentSelectedPortfolio] = React.useState(null);
  const [currentSelectedStock, setCurrentSelectedStock] = React.useState("APPL");

  const sideBarTransitions = {
    open: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5
      }
    },
    closed: {
      opacity: 0.5,
      x: '-100%',
      transition: {
        duration: 1.0
      }
    }
  };

  // const userPortfolios = [{
  //   "userId": 0,
  //   "protfolioId": 0,
  //   "portfolioName": "Default",
  //   "portfolioStockIds": [0, 1, 2],
  // },];

  const handleMessageDialogOpen = () => {
    setMessageDialogOpen(true);
  };

  const handleMessageDialogClose = () => {
    setMessageDialogOpen(false);
  };

  const handleCreatePortfolioDialogOpen = () => {
    setCreatePortfolioDialogOpen(true);
  };

  const handleCreatePortfolioDialogClose = () => {
    setCreatePortfolioDialogOpen(false);
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
          console.log(jsonData.data.length)
          setCompanyData(jsonData.data);
          console.log(jsonData.data[0]);
        } else {
          alert(jsonData.errorMsg);
        }
      }
    }
    catch (err) {
      console.log('Fetch company data failed', err);
    }
    finally {
      //setLoading(false);
    }
  };

  const getUserPortfolio = async (e) => {
    if (props.userData.userId != undefined) {
      const request = {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userId': props.userData.userId,
        })
      }
      try {
        console.log("Try to get user portfolio data");
        //setLoading(true);
        const response = await fetch(BASEURL + "/portfolio", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            setDataLoaded(true);
            console.log(jsonData.data.length)
            jsonData.data.forEach(function(item, index, array){
              item.age = item.age + 1;
            });
            setUserPortfolios(jsonData.data);
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
    }
  };

  React.useEffect(() => {
    if (!dataLoaded) {
      getCompanyData();
      getUserPortfolio();
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
      <CreatePortfolioDialog
        isOpen={isCreatePortfolioDialogOpen}
        handleClose={handleCreatePortfolioDialogClose}
        userPortfolios={userPortfolios}
        setUserPortfolios={setUserPortfolios}
        setCurrentSelectedPortfolio={setCurrentSelectedPortfolio}
        userData={props.userData}
      >
      </CreatePortfolioDialog>
      <PortfolioToolBar
        selectedStocks={selectedStocks}
        setSelectedStocks={setSelectedStocks}
        companyData={companyData}
        setDialogTitle={setDialogTitle}
        setDialogMessage={setDialogMessage}
        isSideBarExpanded={isSideBarExpanded}
        setSideBarExpand={setSideBarExpand}
        openMessageDialog={handleMessageDialogOpen}
        userPortfolios={userPortfolios}
        setUserPortfolios={setUserPortfolios}
        currentSelectedPortfolio={currentSelectedPortfolio}
        setCurrentSelectedPortfolio={setCurrentSelectedPortfolio}
        handleCreatePortfolioDialogOpen={handleCreatePortfolioDialogOpen}
        userData={props.userData}
      >
      </PortfolioToolBar>
      <motion.div
        className={classes.sideBar}
        initial={'closed'}
        animate={isSideBarExpanded ? "open" : "closed"}
        variants={sideBarTransitions}
      >
        <StockSelectSection
          selectedStocks={selectedStocks}
          setSelectedStocks={setSelectedStocks}
          currentSelectedStock={currentSelectedStock}
          setCurrentSelectedStock={setCurrentSelectedStock}
        >
        </StockSelectSection>
      </motion.div>
      <Grid item container direction="row" justify="center" alignItems="stretch" className={classes.portfolioContent}>
        <Grid item xs={6}>
          <Typography className={classes.companyName}  variant="h5">
            Apple Inc.
          </Typography>
          <NewsSection>
          </NewsSection>
        </Grid>
      </Grid>
    </div>
  );
}

export default PortfolioPage;
