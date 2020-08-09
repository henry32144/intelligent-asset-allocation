import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import StockListItem from './StockListItem'
import { FixedSizeList } from 'react-window';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import { Button } from '@material-ui/core';
import CircularProgress from '@material-ui/core/CircularProgress';

const useStyles = makeStyles((theme) => ({
  stockComponent: {
    margin: theme.spacing(0, 0, 2),
  },
  listSubHeader: {
    textAlign: 'initial'
  },
  listTitle: {
    margin: theme.spacing(1, 1, 1),
    display: "inline-flex"
  },
  saveButton: {
    marginLeft: "auto",
    height: "36px",
    width: "64px"
  },
  emptyText: {
    margin: theme.spacing(2, 0, 2),
    textAlign: 'center'
  },
  buttonProgress: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    marginTop: -12,
    marginLeft: -12,
  }
}));


function StockSelectedList(props) {
  const classes = useStyles();

  const saveButtonOnClick = () => {
    props.savePortfolio();
  }

  const removeSelectedStock = (id) => {
    var selectedStocks = Array.from(props.selectedStocks);
    var index = selectedStocks.findIndex(x => x.companyId === id);
    if (index !== -1) {
      selectedStocks.splice(index, 1);
      var newPortfolioStocks = selectedStocks.map(function(item, index, array){
        return item.companySymbol;
      });
      props.setSelectedStocks(newPortfolioStocks);
    }
  };

  const stockListItemOnclick = (symbol) => {
    console.log(symbol);
    if (symbol === props.currentSelectedStock) {
      props.setCurrentSelectedStock(symbol);
    }
  }

  const renderRow = (props) => {
    const { data, index, style } = props;
    const rowItem = data[index];
    return (
      <div style={style}>
        {
          <StockListItem
            companyName={rowItem.companyName}
            companySymbol={rowItem.companySymbol}
            companyId={rowItem.companyId}
            removeSelectedStock={removeSelectedStock}
            stockListItemOnclick={stockListItemOnclick}
          >
          </StockListItem>
        }
      </div>
    );
  };

  return (
    <Box className={classes.stockComponent} component={Grid} container direction="column">
      <Grid item className={classes.listTitle}>
        <Typography variant="h6">
          Stocks
        </Typography>
        <Button 
          className={classes.saveButton} 
          variant="outlined"
          disabled={props.saveButtonLoading}
          onClick={saveButtonOnClick}
        >
          {props.saveButtonLoading ?
            <CircularProgress size={24} className={classes.buttonProgress} />
          :
            "Save"
          }
        </Button>
      </Grid>
      <Divider></Divider>
      { props.selectedStocks.length > 0 ?
      <FixedSizeList
        height={520}
        itemSize={60}
        itemCount={props.selectedStocks.length}
        itemData={props.selectedStocks}
      >
        {renderRow}
      </FixedSizeList >
      :
      <Typography className={classes.emptyText}>
        This portfolio is empty
      </Typography>
      }
    </Box>
  );
}

export default StockSelectedList